module Main (main) where

import System.IO (hSetEncoding, stdout, stderr, utf8)
import System.Random (newStdGen)
import Types
import Environment
import DQN
import Render

main :: IO ()
main = do
  hSetEncoding stdout utf8
  hSetEncoding stderr utf8

  renderHeader
  putStrLn "Initializing Dungeon AI..."
  putStrLn ""

  let config = defaultDQNConfig
  putStrLn "Configuration:"
  putStrLn $ "   Episodes:      " ++ show (dqnNumEpisodes config)
  putStrLn $ "   Max Steps:     " ++ show (dqnMaxSteps config)
  putStrLn $ "   Hidden Size:   " ++ show (dqnHiddenSize config)
  putStrLn $ "   Input Size:    " ++ show (dqnInputSize config)
  putStrLn $ "   Output Size:   " ++ show (dqnOutputSize config)
  putStrLn $ "   Gamma:         " ++ show (acGamma (dqnAgentConfig config))
  putStrLn $ "   Learning Rate: " ++ show (acLearningRate (dqnAgentConfig config))
  putStrLn $ "   Buffer Size:   " ++ show (dqnBufferSize config)
  putStrLn ""

  let env = mkGridWorld
      syms = defaultSymbols
  putStrLn "Initial Grid World:"
  putStrLn $ renderGrid syms env
  putStrLn "Training the agent..."
  putStrLn ""
  putStrLn "------+---------------------------+--------+---------+--------------+--------"
  putStrLn "  Ep  | Reward                    | Steps  | Epsilon | Loss         | Goal   "
  putStrLn "------+---------------------------+--------+---------+--------------+--------"

  gen <- newStdGen
  let (trainedNet, stats) = trainDQN config gen

  mapM_ (\s -> do
    let ep = tsEpisode s
    if ep <= 10 || ep `mod` 25 == 0 || ep == dqnNumEpisodes config
      then putStrLn (renderTrainingStats s)
      else return ()
    ) stats

  putStrLn "------+---------------------------+--------+---------+--------------+--------"

  renderFinalSummary stats
  renderLearnedPath syms trainedNet
  putStrLn ""
  putStrLn "Dungeon AI training complete!"
  putStrLn ""
