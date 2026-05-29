module Render
  ( renderGrid
  , renderGridWithPath
  , renderTrainingStats
  , renderHeader
  , renderFinalSummary
  , renderLearnedPath
  ) where

import Numeric.LinearAlgebra (toList)
import Types
import Environment
import NeuralNetwork (predict)

renderHeader :: IO ()
renderHeader = do
  putStrLn ""
  putStrLn "╔══════════════════════════════════════════════════════╗"
  putStrLn "║           🏰  DUNGEON AI  🏰                       ║"
  putStrLn "║       Deep Q-Learning Grid World Agent              ║"
  putStrLn "║                                                     ║"
  putStrLn "║   🧙 Aventureiro   💎 Tesouro   🔥 Armadilha       ║"
  putStrLn "╚══════════════════════════════════════════════════════╝"
  putStrLn ""

renderGrid :: Symbols -> GridWorld -> String
renderGrid syms gw =
  let (rows, cols) = gwSize gw
      border = "  ┌" ++ concat (replicate cols "────") ++ "─┐"
      bottom = "  └" ++ concat (replicate cols "────") ++ "─┘"
      header = "    " ++ concatMap (\c -> " " ++ show c ++ "  ") [0 .. cols - 1]
      gridLines = map (renderRow syms gw cols) [0 .. rows - 1]
  in unlines (header : border : gridLines ++ [bottom])

renderRow :: Symbols -> GridWorld -> Int -> Int -> String
renderRow syms gw cols row =
  let cells = map (\col -> cellSymbol syms gw (row, col)) [0 .. cols - 1]
  in show row ++ " │ " ++ unwords cells ++ " │"

cellSymbol :: Symbols -> GridWorld -> Position -> String
cellSymbol syms gw pos
  | pos == gwAgentPos gw = symAgent syms
  | pos == gwGoalPos gw  = symGoal syms
  | pos `elem` gwObstacles gw = symObstacle syms
  | otherwise            = symEmpty syms

renderGridWithPath :: Symbols -> GridWorld -> [Position] -> String
renderGridWithPath syms gw path =
  let (rows, cols) = gwSize gw
      border = "  ┌" ++ concat (replicate cols "────") ++ "─┐"
      bottom = "  └" ++ concat (replicate cols "────") ++ "─┘"
      header = "    " ++ concatMap (\c -> " " ++ show c ++ "  ") [0 .. cols - 1]
      gridLines = map (renderRowWithPath syms gw cols path) [0 .. rows - 1]
  in unlines (header : border : gridLines ++ [bottom])

renderRowWithPath :: Symbols -> GridWorld -> Int -> [Position] -> Int -> String
renderRowWithPath syms gw cols path row =
  let cells = map (\col -> cellSymbolWithPath syms gw path (row, col)) [0 .. cols - 1]
  in show row ++ " │ " ++ unwords cells ++ " │"

cellSymbolWithPath :: Symbols -> GridWorld -> [Position] -> Position -> String
cellSymbolWithPath syms gw path pos
  | pos == gwGoalPos gw             = symGoal syms
  | pos `elem` gwObstacles gw      = symObstacle syms
  | pos `elem` path                 = symPath syms
  | pos == gwStartPos gw            = symAgent syms
  | otherwise                       = symEmpty syms

renderTrainingStats :: TrainingStats -> String
renderTrainingStats stats =
  let ep      = tsEpisode stats
      reward  = tsTotalReward stats
      steps   = tsSteps stats
      eps     = tsEpsilon stats
      loss    = tsAverageLoss stats
      goal    = if tsReachedGoal stats then "+" else "-"
      bar     = rewardBar reward
  in concat
    [ "| Ep "
    , padLeft 4 (show ep)
    , " | Reward: "
    , padLeft 8 (showF1 reward)
    , " "
    , bar
    , " | Steps: "
    , padLeft 4 (show steps)
    , " | eps: "
    , padLeft 6 (showF3 eps)
    , " | Loss: "
    , padLeft 10 (showF4 loss)
    , " | Goal: "
    , goal
    , " |"
    ]

rewardBar :: Double -> String
rewardBar r =
  let normalized = max 0 (min 10 (round ((r + 100) / 20) :: Int))
      filled     = replicate normalized '#'
      empty'     = replicate (10 - normalized) '.'
  in "[" ++ filled ++ empty' ++ "]"

renderFinalSummary :: [TrainingStats] -> IO ()
renderFinalSummary stats = do
  let totalEps    = length stats
      goalReached = length (filter tsReachedGoal stats)
      successRate = (fromIntegral goalReached / fromIntegral totalEps * 100) :: Double
      last50      = take 50 (reverse stats)
      goalLast50  = length (filter tsReachedGoal last50)
      rateLast50  = (fromIntegral goalLast50 / fromIntegral (length last50) * 100) :: Double
      avgReward   = sum (map tsTotalReward stats) / fromIntegral totalEps
      bestReward  = maximum (map tsTotalReward stats)
  putStrLn ""
  putStrLn "========================================================="
  putStrLn "                  TRAINING SUMMARY                        "
  putStrLn "========================================================="
  putStrLn $ "  Total Episodes:      " ++ show totalEps
  putStrLn $ "  Goals Reached:       " ++ show goalReached
  putStrLn $ "  Success Rate:        " ++ showF1 successRate ++ "%"
  putStrLn $ "  Last 50 Success:     " ++ showF1 rateLast50 ++ "%"
  putStrLn $ "  Avg Reward:          " ++ showF1 avgReward
  putStrLn $ "  Best Reward:         " ++ showF1 bestReward
  putStrLn "========================================================="

renderLearnedPath :: Symbols -> Network -> IO ()
renderLearnedPath syms net = do
  putStrLn ""
  putStrLn "========================================================="
  putStrLn "              LEARNED PATH                                "
  putStrLn "========================================================="
  let env0 = mkGridWorld
      path = tracePath net env0 [] 50
  putStrLn $ renderGridWithPath syms (resetEnv env0) path
  putStrLn $ "  " ++ symAgent syms ++ " = Agent  " ++ symGoal syms ++ " = Goal  " ++ symObstacle syms ++ " = Trap  " ++ symPath syms ++ " = Path"
  putStrLn "========================================================="

tracePath :: Network -> GridWorld -> [Position] -> Int -> [Position]
tracePath _ _ acc 0 = acc
tracePath net env acc remaining
  | gwDone env = acc
  | otherwise  =
      let state     = stateToVector env
          qValues   = predict net state
          qList     = toList qValues
          bestIdx   = argmax qList
          action    = indexToAction bestIdx
          (env', _, _) = stepEnv env action
          newPos    = gwAgentPos env'
      in if newPos `elem` acc
         then acc
         else tracePath net env' (acc ++ [newPos]) (remaining - 1)

argmax :: [Double] -> Int
argmax []     = 0
argmax xs     = snd $ foldl1 (\(v1, i1) (v2, i2) -> if v2 > v1 then (v2, i2) else (v1, i1)) (zip xs [0..])

padLeft :: Int -> String -> String
padLeft n s = replicate (max 0 (n - length s)) ' ' ++ s

showF1 :: Double -> String
showF1 x =
  let whole = truncate x :: Int
      frac  = abs (round ((x - fromIntegral whole) * 10)) :: Int
  in show whole ++ "." ++ show (min 9 frac)

showF3 :: Double -> String
showF3 x =
  let whole = truncate x :: Int
      frac  = abs (round ((x - fromIntegral whole) * 1000)) :: Int
      fracStr = padLeft 3 (show frac)
      fracStr' = map (\c -> if c == ' ' then '0' else c) fracStr
  in show whole ++ "." ++ fracStr'

showF4 :: Double -> String
showF4 x =
  let whole = truncate x :: Int
      frac  = abs (round ((x - fromIntegral whole) * 10000)) :: Int
      fracStr = padLeft 4 (show frac)
      fracStr' = map (\c -> if c == ' ' then '0' else c) fracStr
  in show whole ++ "." ++ fracStr'
