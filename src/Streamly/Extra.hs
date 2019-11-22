{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE NumericUnderscores #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE OverloadedLabels #-}

module Streamly.Extra where

import           GHC.Natural
import           Control.Arrow
import           Control.Concurrent hiding (yield)
import qualified Control.Concurrent.STM.TChan as TChan
import qualified Control.Concurrent.STM.TBQueue as BQ
import           Control.Monad
import           Control.Monad.Catch (MonadMask)
import           Control.Monad.Except (catchError, MonadError)
import           Control.Monad.IO.Class
import           Control.Monad.Reader.Class
import qualified Control.Monad.STM as STM
import qualified Data.ByteString as BS
import           Data.Function
import           Data.Functor
import           Data.Either
import qualified Data.Internal.SortedSet as ZSet
import           Data.IORef
import qualified Data.Map.Strict as Map
import           Data.Maybe (fromMaybe, isJust)
import qualified Data.Set as Set
import           Network.Socket ( Socket(..), close)
import           Network.Socket.ByteString (recv)
import qualified Streamly as S
import qualified Streamly.Internal.Data.Fold as FL
import qualified Streamly.Prelude as SP
import qualified Streamly.Internal.Data.Pipe as Pipe

-- | Group the stream into a smaller set of keys and fold elements of a specific key
demuxByM
  :: Eq b
  => Ord b
  => Monad m
  => (a -> m b)
  -> FL.Fold m a c
  -> FL.Fold m a [(b, c)]
demuxByM f (FL.Fold step' begin' done')
  = FL.Fold step begin done
  where
    begin = pure mempty
    step hm a = do
      b <- f a
      (\c -> Map.insert b c hm) <$>
        (maybe begin' pure (Map.lookup b hm) >>= flip step' a)
    done = fmap Map.toList <<< mapM done'

demuxAndAggregateByInterval
  :: Eq b
  => Ord b
  => S.MonadAsync m
  => S.IsStream t
  => (a -> m b)
  -> Double
  -> FL.Fold m a c
  -> t m a
  -> t m [(b, c)]
demuxAndAggregateByInterval f delay agg =
  SP.intervalsOf delay (demuxByM f agg)

demuxAndAggregateInChunks
  :: Eq b
  => Ord b
  => S.MonadAsync m
  => S.IsStream t
  => (a -> m b)
  -> Int
  -> FL.Fold m a c
  -> t m a
  -> t m [(b, c)]
demuxAndAggregateInChunks f chunkSize agg =
  SP.chunksOf chunkSize (demuxByM f agg)

demuxByAndAggregateInChunksOf
  :: Eq b
  => Ord b
  => Show b
  => S.MonadAsync m
  => S.IsStream t
  => (a -> m b)
  -> Int
  -> FL.Fold m a c
  -> t m a
  -> t m (b, c)
demuxByAndAggregateInChunksOf f i (FL.Fold step' begin' done') src
  = SP.mapMaybe id $ SP.scan (FL.Fold step begin done) src
  where
  begin = pure (mempty, Nothing)
  step (hm, _) a = do
    b <- f a
    (j, x) <- maybe ((1, ) <$> begin') pure (Map.lookup b hm)
    (\x' -> if j < i
      then (Map.insert b (j+1, x') hm, Nothing)
      else (Map.delete b hm, Just (b, x'))) <$> step' x a
  done (_, Just (b, x')) = Just . (b, ) <$> done' x'
  done _ = pure Nothing

-- | Collects elements from the stream into a key given by the `keyFn`
--   Once the stream for that key is completed or on a configurable timeout, it is available on the output stream.
collectTillEndOrTimeout
  :: Eq b
  => S.MonadAsync m
  => Ord b
  => (a -> b)
  -> ([a] -> Bool)
  -> Int
  -> S.SerialT m a
  -> S.SerialT m (Maybe [a])
collectTillEndOrTimeout keyFn isEnd timeout src =
  liftIO TChan.newTChanIO >>= \chan ->
    completedSessionStream chan `S.parallel` incompletedSessionsStream chan
  where
    completedSessionStream c =
      SP.scan
        (FL.Fold
          -- State is (IORef (HM b (ThreadId, [a])), IORef (Maybe [a]))
          -- First HM is a IORef because it should be modifiable by another thread
          -- Second is a IORef because every time an extract is called, it has to be cleared
          (\(hmRef, outRef) a -> liftIO $ do
            let b = keyFn a
            mTIdAndLogs <- Map.lookup b <$> readIORef hmRef
            case mTIdAndLogs of
              Nothing ->
                if isEnd [a]
                then
                  (hmRef, outRef) <$ atomicWriteIORef outRef (Just [a])
                else do
                  tId <- liftIO $
                    forkIO $
                      threadDelay (timeout * 1000000)
                      *> atomicModifyIORef' hmRef (\hm -> (Map.delete b hm, snd <$> Map.lookup b hm))
                        >>= STM.atomically . TChan.writeTChan c
                  atomicModifyIORef' hmRef (\hm -> (Map.insert b (tId, [a]) hm, ()))
                  atomicWriteIORef outRef Nothing
                  pure (hmRef, outRef)
              Just (tId, as) ->
                let as' = a : as
                in (hmRef, outRef) <$
                  if isEnd as'
                    then
                      killThread tId
                      *> atomicWriteIORef outRef (Just as')
                      *> atomicModifyIORef' hmRef (\hm -> (Map.delete b hm, ()))
                    else
                      atomicWriteIORef outRef Nothing
                      *> atomicModifyIORef' hmRef (\hm -> (Map.insert b (tId, as') hm, ()))
          )
          ((,) <$> liftIO (newIORef mempty) <*> liftIO (newIORef mempty))
          (\(_, outRef) -> liftIO $ atomicModifyIORef' outRef (\mAs -> (Nothing, mAs))))
        src
    incompletedSessionsStream c = SP.repeatM (liftIO $ STM.atomically $ TChan.readTChan c)

-- Reads lines from a socket and produces a parsed stream
lineStream
  :: (S.MonadAsync m, MonadMask m, MonadError e m)
  => Socket
  -> (BS.ByteString -> (Maybe [b], BS.ByteString))
  -> S.SerialT m b
lineStream sock parser =
  SP.concatMap SP.fromList
  . fmap (fromMaybe mempty)
  . SP.takeWhile isJust $
    fst <$>
      S.maxBuffer (-1)
        (SP.iterateM (\(_, buf) -> do
          b <- catchError (liftIO . recv sock $ 4096) (\_ -> pure mempty)
          if BS.null b
            then (liftIO . close $ sock) $> (Nothing, b)
          else buf <> b & pure . parser)
          (pure (Just mempty, mempty)))

runAllWith
  :: S.MonadAsync m
  => S.IsStream t
  => MonadIO (t m)
  => (forall c. t m c -> t m c -> t m c)
  -> [ t m a -> t m () ]
  -> t m a
  -> t m ()
runAllWith combine fs src = do
  writeChan <- liftIO TChan.newBroadcastTChanIO
  SP.mapM (liftIO . STM.atomically . TChan.writeTChan writeChan) src
    `S.parallel`
    S.forEachWith combine fs (\f -> do
      chan' <- liftIO $ STM.atomically $ TChan.dupTChan writeChan
      f $ SP.repeatM $ liftIO $
        STM.atomically $
          TChan.readTChan chan')

-- Works only on infinite streams.
duplicate
  :: S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => t m a
  -> m (t m a, t m a)
duplicate src = do
  (writeChan, readChan1, readChan2) <- liftIO $ do
    chan <- TChan.newBroadcastTChanIO
    chan' <- STM.atomically $ TChan.dupTChan chan
    chan'' <- STM.atomically $ TChan.dupTChan chan
    pure (chan, chan', chan'')
  let
    writes =
      SP.mapM (liftIO . STM.atomically . TChan.writeTChan writeChan) src
    reads1 =
      SP.repeatM (liftIO $ STM.atomically $ TChan.readTChan readChan1)
    reads2 =
      SP.repeatM (liftIO $ STM.atomically $ TChan.readTChan readChan2)
  pure (fmap (fromRight undefined) $ SP.filter isRight $ (Left <$> writes) `S.async` (Right <$> reads1), reads2)

tap
  :: S.MonadAsync m
  => S.IsStream t
  => t m a
  -> (a -> m b)
  -> t m a
tap s f = SP.mapM (\x -> x <$ f x) s

(|>>)
  :: S.MonadAsync m
  => S.IsStream t
  => t m a
  -> (a -> m b)
  -> t m a
(|>>) = tap

infixl 5 |>>

firstOcc
  :: Ord a
  => Monad m
  => S.IsStream t
  => t m a
  -> t m a
firstOcc =
  SP.mapMaybe id
  .  SP.scan (FL.Fold step begin end)
  where
  step (x, _) a =
    pure (Set.insert a x, if Set.member a x then Nothing else Just a)
  begin =
    pure (Set.empty, Nothing)
  end = snd >>> pure

-- | Stream which samples from the latest value from the first stream at times when the second stream yields
--   Note : Doesn't produce values until one value is yield'ed from each stream
--   everyNSecsIncBy n i =
--     SP.iterateM (\j -> threadDelay (n * 1000000) >> pure (i + j)) (pure 0)
--   everyNSecondsAddOrSub n =
--     snd <$>
--       SP.iterateM
--         (\(b, _) -> threadDelay (n * 1000000) >> pure (if b then (False, (\x -> (x,2+x))) else (True, (\x -> (x,x-2)) )))
--         (pure (True, (\x -> (x,x-2)) ))
--   SP.mapM_ print $ sampleOn (everyNSecsIncBy 1 2) (everyNSecondsAddOrSub 4)
--   outputs :
--   (0,-2)
--   (6,8)
--   (14,12)
--   (22,24)
--   (30,28)
--   (38,40)
--   (46,44)
--   (54,56)
--   (62,60)
sampleOn
  :: S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => t m a
  -> t m (a -> b)
  -> t m b
sampleOn src pulse =
  SP.mapMaybe id $
    SP.scan fld combined
  where
  combined =
    (Left <$> src) `S.parallel` (Right <$> pulse)
  fld = FL.Fold step begin done
  -- First is the latest value of source,
  -- second is the value which to be yield'ed
  step _ (Left a) = pure (Just a, Nothing)
  step (x, _) (Right f) = pure (x, f <$> x)
  begin = pure (Nothing, Nothing)
  done (_, out) = pure out

applyWithLatestM
  :: S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => (a -> b -> m c)
  -> t m a
  -> t m b
  -> t m c
applyWithLatestM f s1 s2 =
  SP.mapMaybe id $
    SP.scan fld combined
  where
  combined =
    (Left <$> s1) `S.parallel` (Right <$> s2)
  fld = FL.Fold step begin done
  begin = pure (Nothing, Nothing)
  step (Just b, _) (Left !a) = (Just b,) . Just <$> f a b
  step (Nothing, _) (Left !a) = pure (Nothing, Nothing)
  step _ (Right !b) = pure (Just b, Nothing)
  done (_, !out) = pure out
-- | Stream which produces values as fast as the faster stream(the first argument)
--   using the latest value from the slower stream(the second argument)
--   Note : Doesn't produce values until one value is yield'ed from each stream
--   everyNSecsIncBy n i =
--     SP.iterateM
--       (\j -> threadDelay (n * 1000000) >> pure (i + j))
--       (pure 0)
--   everyNSecondsAddOrSub n =
--     snd <$>
--       SP.iterateM
--         (\(b, _) -> threadDelay (n * 1000000) >> pure (if b then (False, (\x -> (x,2+x))) else (True, (\x -> (x,x-2)) )))
--         (pure (True, (\x -> (x,x-2)) ))
--   SP.mapM_ print $ applyWithLatest (everyNSecsIncBy 1 2) (everyNSecondsAddOrSub 4)
--   outputs :
--   (2,0)
--   (4,2)
--   (6,4)
--   (8,10)
--   (10,12)
--   (12,14)
--   (14,16)
--   (16,14)
--   (18,16)
--   (20,18)
--   (22,20)
--   (24,26)
--   (26,28)
--   (28,30)
--   (30,32)
--   (32,30)
applyWithLatest
  :: S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => t m a
  -> t m (a -> b)
  -> t m b
applyWithLatest =
  applyWithLatestM (\a f -> pure $ f a)

-- | Stream which races a function stream and a argument stream
--   and uses the latest value of the other stream whenever any of the stream yields a value
--   Note : Doesn't produce values until one value is yield'ed from each stream
--   everyNSecsIncBy n i =
--     SP.iterateM
--       (\j -> threadDelay (n * 1000000) >> pure (i + j))
--       (pure 0)
--   everyNSecondsAddOrSub n =
--     snd <$>
--       SP.iterateM
--         (\(b, _) -> threadDelay (n * 1000000) >> pure (if b then (False, (\x -> (x,2+x))) else (True, (\x -> (x,x-2)) )))
--         (pure (True, (\x -> (x,x-2)) ))
--   SP.mapM_ print $ zipAsyncly' (everyNSecsIncBy 4 2)  (everyNSecondsAddOrSub 1)
--   outputs :
--   (0,-2)
--   (0,2)
--   (0,-2)
--   (0,2)
--   (2,4)
--   (2,0)
--   (2,4)
--   (2,0)
--   (2,4)
--   (4,6)
--   (4,2)
--   (4,6)
--   (4,2)
--   (4,6)
--   (6,8)
--   (6,4)
--   (6,8)
--   (6,4)
--   (6,8)
--   (8,10)
--   (8,6)
--   (8,10)
--   SP.mapM_ print $ zipAsyncly' (everyNSecsIncBy 1 2)  (everyNSecondsAddOrSub 4)
--   (0,-2)
--   (2,0)
--   (4,2)
--   (6,4)
--   (6,8)
--   (8,10)
--   (10,12)
--   (12,14)
--   (14,16)
--   (14,12)
--   (16,14)
--   (18,16)
--   (20,18)
--   (22,20)
--   (22,24)
--   (24,26)
--   (26,28)
--   (28,30)
--   (30,32)
zipAsyncly'
  :: S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => t m a
  -> t m (a -> b)
  -> t m b
zipAsyncly' aSrc fSrc =
  SP.mapMaybe id $
    SP.scan fld combined
  where
  combined =
    (Left <$> aSrc) `S.parallel` (Right <$> fSrc)
  fld = FL.Fold step begin done
  -- First is the latest value of a -> b,
  -- Second is the latest value of a,
  -- Third is the value which to be yield'ed
  begin = pure (Nothing, Nothing, Nothing)
  step (maybeF, _, _) (Left a) =
    pure $
      maybe
        (Nothing, Just a, Nothing)
        (\f -> (Just f, Just a, Just (f a)))
        maybeF
  step (_, maybeA, _) (Right f) =
    pure $
      maybe
        (Just f, Nothing, Nothing)
        (\a -> (Just f, Just a, Just (f a)))
        maybeA
  done (_, _, out) = pure out

-- | Group incoming elements into buckets of @tickInterval × timeThreshold@
--   microseconds and output only the first occurrence of each element.
--   This will yield "1" every five seconds:
--   >>> num1Every1MilliSec = SP.repeatM (threadDelay 1000 *> pure 1)
--   >>> SP.mapM_ print $ firstOccWithin 1000000 5 num1Every1MilliSec
--   New elements will be yielded only once per @tickInterval@, so choose it
--   depending on the needed granularity.
firstOccWithin
  :: Ord a
  => S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => Int
  -> Int
  -> t m a
  -> t m a
firstOccWithin tickInterval timeThreshold src
  =
  SP.mapMaybe id $
    SP.scan
      (FL.Fold step begin end)
      srcWithTicker
  where
  step (x, _) (a, (up, down)) =
    pure (if ZSet.zMember a newX then (newX, Nothing) else (ZSet.zAdd a up newX, Just a))
    where
      newX = if down == 0 then ZSet.zRangeGTByScore (up - timeThreshold) x else x
  begin =
    pure (ZSet.zempty, Nothing)
  end = pure . snd

  ticker =
    SP.mapM (<$ liftIO (threadDelay tickInterval)) $
    SP.fromFoldable $
    zip [0..] (cycle [timeThreshold, timeThreshold - 1 .. 0])

  srcWithTicker =
    src `applyWithLatest` ((\i a -> (a, i)) <$> ticker)

groupConsecutiveBy
  :: Eq b
  => Monad m
  => (a -> b)
  -> FL.Fold m (Maybe a) (Maybe [a])
groupConsecutiveBy f = FL.Fold step begin end
  where
  -- State is a tuple of 3 elements
  -- First is the optional Last Id we have seen.
  -- Second is the accumulated a's for the identifier represented by the first element
  -- Third is a Maybe [a], if Just xs then it is a completed set of a's
  -- Else if Nothing, it means that the set of a's seen till now is not completed
  begin = pure (Nothing, [], Nothing)
  end (_, _, maybeXS) = pure maybeXS
  step (Just oldId, oldXS, _) (Just newElem)
    | oldId == f newElem = pure (Just oldId, newElem : oldXS, Nothing)
    | otherwise = pure (Just (f newElem), [newElem], Just oldXS)
  step (Just _, oldXS, _) Nothing = pure (Nothing, [], Just oldXS)
  step (Nothing, _, _) maybeNewElem =
    maybe
      (pure (Nothing, [], Nothing))
      (\newElem -> pure (Just (f newElem), [newElem], Nothing))
      maybeNewElem

counts
  :: Applicative m
  => Ord a
  => FL.Fold m a (Map.Map a Int)
counts = FL.Fold step begin end
  where
  step x a =
    pure $ Map.alter (Just . maybe 1 (+1)) a x
  begin = pure mempty
  end = pure

data Direction = IN | OUT deriving (Show, Eq, Ord)
type Logger tag = tag -> Direction -> Int -> IO ()
data LoggerConfig tag
  = LoggerConfig
  { logger :: Logger tag
  , samplingRate :: Double {-Rate of measurement in seconds-}}

withRateGaugeWithElements
  :: forall t m a b tag
  . Applicative m
  => S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => MonadReader (LoggerConfig tag) (t m)
  => Ord tag
  => (a -> tag)
  -> t m a
  -> t m a
withRateGaugeWithElements tagGenerator src =
  ask >>= measureAndRecord IN
  where
  measureAndRecord :: Direction -> LoggerConfig tag -> t m a
  measureAndRecord direction (LoggerConfig { logger, samplingRate }) =
    SP.mapMaybe id $
      SP.scan (FL.Fold step begin end) withTimer
    where
    step (counts, _) Nothing =
      (Map.map (const 0) counts, Nothing) <$
        Map.traverseWithKey (\tag count -> liftIO $ logger tag direction count) counts
    step (counts, _) (Just a) =
      pure (Map.alter (Just . maybe 1 (+1)) (tagGenerator a) counts, Just a)
    begin = pure (Map.empty, Nothing)
    end = pure . snd
    withTimer = (Just <$> src) `S.parallel` (Nothing <$ timeout)
    timeout = SP.repeatM $ liftIO $ threadDelay (fromEnum (samplingRate * 1_000_000))

withRateGauge
  :: forall t m a b tag
  . Applicative m
  => S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => MonadReader (LoggerConfig tag) (t m)
  => Ord tag
  => tag
  -> t m a
  -> t m a
withRateGauge tag = withRateGaugeWithElements (const tag)

withThroughputGauge
  :: forall t m a b tag
  . Applicative m
  => S.MonadAsync m
  => S.IsStream t
  => Monad (t m)
  => tag
  -> Logger tag
  -> (t m a -> t m b)
  -> t m a
  -> t m b
withThroughputGauge tag recordMeasurement f =
  measureAndRecord OUT . f . measureAndRecord IN
  where
  measureAndRecord :: Direction -> t m c -> t m c
  measureAndRecord direction src =
    SP.mapMaybe id $
      SP.scan (FL.Fold step begin end) withTimer
    where
    step (count, _) Nothing = liftIO (recordMeasurement tag direction count) $> (0, Nothing)
    step (count, _) (Just a) = pure (count + 1, Just a)
    begin = pure (0, Nothing)
    end = pure . snd
    withTimer = (Just <$> src) `S.parallel` (Nothing <$ timeout)
    timeout = SP.repeatM $ liftIO $ threadDelay 1000000

consumeWithBuffer
  :: forall t m a b
  . Applicative m
  => S.IsStream t
  => MonadIO (t m)
  => S.MonadAsync m
  => Natural
  -> (a -> m ())
  -> t m a
  -> t m ()
consumeWithBuffer n drainBy src =
  liftIO (BQ.newTBQueueIO n) >>= \chan -> producer chan `S.parallel` consumer chan
  where
    producer chan =
      SP.mapM
        (\a -> liftIO $ STM.atomically $ do
          b <- BQ.isFullTBQueue chan
          if b then pure () else BQ.writeTBQueue chan a)
        src
    consumer :: BQ.TBQueue a -> t m ()
    consumer chan =
      SP.mapM drainBy $
        SP.repeatM (liftIO $ STM.atomically (BQ.readTBQueue chan))
