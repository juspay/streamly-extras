--  With help from https://jameshfisher.com/2018/04/22/redis-sorted-set/
module Data.Internal.SortedSet where

import Prelude
import Data.Map (Map)
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Data.Set (Set)

data ZSet k v = ZSet
  { scores  :: !(Map k v)
  , byScore :: !(Map v (Set k)) } deriving Show

type MultiMap k v = Map.Map k (Set v)

zempty :: ZSet k v
zempty = ZSet Map.empty Map.empty

zAdd :: (Ord k, Ord v) => k -> v -> ZSet k v -> ZSet k v
zAdd x newScore z = ZSet (Map.insert x newScore (scores z)) newByScore where
  newByScore = Map.alter (Just . maybe (Set.singleton x) (Set.insert x)) newScore oldScoreRemoved
  oldScoreRemoved = maybe (byScore z) (\oldScore -> multiMapDelete oldScore x (byScore z)) (Map.lookup x (scores z))

multiMapDelete :: (Ord k, Ord v) => k -> v -> Map k (Set v) -> Map k (Set v)
multiMapDelete k v = Map.alter f k where
  f Nothing = Nothing
  f (Just vs) = let vs' = Set.delete v vs
                in if Set.null vs' then Nothing else Just vs'

zRangeGTByScore :: (Ord k, Ord v) => v -> ZSet k v -> ZSet k v
zRangeGTByScore maxScore (ZSet scores byScore) =
  ZSet
    (Map.restrictKeys scores $ Set.unions $ Map.elems newByScore)
    newByScore
  where
  newByScore = snd $ Map.split maxScore byScore

zMember :: Ord k => k -> ZSet k v -> Bool
zMember k = Map.member k . scores
