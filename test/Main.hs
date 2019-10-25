{-# LANGUAGE FlexibleContexts #-}
import Prelude
import Control.Concurrent
import qualified Streamly as S
import Streamly.Extra
import qualified Streamly.Prelude as SP
import Data.Functor
import Data.List
import Control.Monad.IO.Class
import Control.Monad.Trans.Control
import Control.Monad.Reader
import Data.Maybe

type Count = Int
type Duration = Int
type Rate = (Count, Duration)

src
  :: MonadIO m
  => S.MonadAsync m
  => Rate -> S.SerialT m Int
src (count, duration) =
  SP.replicateM 10 (liftIO (threadDelay duration) $> [1..count]) >>= SP.fromList

stdErrLogger :: Logger String
stdErrLogger tag dir i = putStrLn $ tag <> " " <> acting <> " at the rate of " <> show i <> " records /sec"
  where
  acting = if dir == IN  then "consuming" else "producing"

main :: IO ()
main =
  runReaderT
    (SP.drainWhile isJust pipeline)
    (LoggerConfig stdErrLogger 5.0)
  where
  pipeline = (Just <$> (triple . filterEven) mainStream) <> pure Nothing
  mainStream =
    SP.mapMaybe id $ SP.takeWhile isJust $ withRateGauge "src" $ (Just <$> src (100,1000000)) <> pure Nothing
  filterEven = withRateGauge "filter" . SP.filter even
  triple = withRateGauge "replicate" . SP.concatMap (\a -> SP.fromList [a,a,a])
