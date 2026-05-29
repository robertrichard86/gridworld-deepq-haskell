module Environment
  ( mkGridWorld
  , resetEnv
  , stepEnv
  , stateToVector
  , isValidPosition
  , cellAt
  , stateSize
  ) where

import Numeric.LinearAlgebra (Vector, fromList)
import Types

mkGridWorld :: GridWorld
mkGridWorld = GridWorld
  { gwSize      = (5, 5)
  , gwAgentPos  = (0, 0)
  , gwGoalPos   = (4, 4)
  , gwObstacles = [(0, 3), (1, 1), (2, 3), (3, 1)]
  , gwStartPos  = (0, 0)
  , gwDone      = False
  }

resetEnv :: GridWorld -> GridWorld
resetEnv gw = gw
  { gwAgentPos = gwStartPos gw
  , gwDone     = False
  }

stepEnv :: GridWorld -> Action -> (GridWorld, Double, Bool)
stepEnv gw action
  | gwDone gw = (gw, 0.0, True)
  | otherwise =
      let (row, col)    = gwAgentPos gw
          (rows, cols)  = gwSize gw
          (newRow, newCol) = case action of
            Up        -> (row - 1, col)
            Down      -> (row + 1, col)
            MoveLeft  -> (row, col - 1)
            MoveRight -> (row, col + 1)
          clampedPos = (clamp 0 (rows - 1) newRow, clamp 0 (cols - 1) newCol)
          hitWall    = clampedPos == (row, col) && (newRow, newCol) /= (row, col)
          finalPos   = if hitWall then (row, col) else clampedPos
          isGoal     = finalPos == gwGoalPos gw
          isObs      = finalPos `elem` gwObstacles gw
          reward
            | isGoal    =  100.0
            | isObs     = -100.0
            | hitWall   =   -2.0
            | otherwise =   -1.0
          done = isGoal || isObs
          newGW = gw
            { gwAgentPos = finalPos
            , gwDone     = done
            }
      in (newGW, reward, done)

clamp :: Int -> Int -> Int -> Int
clamp lo hi x = max lo (min hi x)

isValidPosition :: GridWorld -> Position -> Bool
isValidPosition gw (r, c) =
  let (rows, cols) = gwSize gw
  in r >= 0 && r < rows && c >= 0 && c < cols

cellAt :: GridWorld -> Position -> CellType
cellAt gw pos
  | pos == gwGoalPos gw         = Goal
  | pos `elem` gwObstacles gw  = Obstacle
  | pos == gwStartPos gw        = Start
  | otherwise                   = Empty

stateSize :: Int
stateSize = 12

stateToVector :: GridWorld -> Vector Double
stateToVector gw =
  let (rows, cols)     = gwSize gw
      (agentR, agentC) = gwAgentPos gw
      (goalR, goalC)   = gwGoalPos gw
      normR r = fromIntegral r / fromIntegral (rows - 1)
      normC c = fromIntegral c / fromIntegral (cols - 1)
      agentFeatures = [normR agentR, normC agentC]
      goalFeatures  = [normR goalR, normC goalC]
      distFeatures  = [ normR (goalR - agentR)
                       , normC (goalC - agentC)
                       ]
      obstacleFeatures = concatMap (\(or', oc) ->
        [ normR (or' - agentR)
        , normC (oc - agentC)
        ]) (take 3 (gwObstacles gw))
      padding = replicate (6 - length obstacleFeatures) 0.0
  in fromList (agentFeatures ++ goalFeatures ++ distFeatures ++ obstacleFeatures ++ padding)
