module Main (main) where

import Test.Hspec
import Numeric.LinearAlgebra (fromList, toList, size)
import System.Random (mkStdGen)

import Types
import Environment
import NeuralNetwork
import ReplayBuffer
import Agent

main :: IO ()
main = hspec $ do

  describe "Types" $ do
    it "allActions contains exactly 4 actions" $
      length allActions `shouldBe` 4

    it "actionToIndex and indexToAction are inverses" $
      map (indexToAction . actionToIndex) allActions `shouldBe` allActions

    it "numActions equals 4" $
      numActions `shouldBe` 4

  describe "Environment" $ do
    it "creates a 5x5 grid world" $ do
      let gw = mkGridWorld
      gwSize gw `shouldBe` (5, 5)

    it "starts agent at position (0,0)" $ do
      let gw = mkGridWorld
      gwAgentPos gw `shouldBe` (0, 0)

    it "places goal at position (4,4)" $ do
      let gw = mkGridWorld
      gwGoalPos gw `shouldBe` (4, 4)

    it "has 4 obstacles" $ do
      let gw = mkGridWorld
      length (gwObstacles gw) `shouldBe` 4

    it "resets environment correctly" $ do
      let gw = mkGridWorld
          (gw', _, _) = stepEnv gw Down
          gw'' = resetEnv gw'
      gwAgentPos gw'' `shouldBe` (0, 0)
      gwDone gw'' `shouldBe` False

    it "moves agent down correctly" $ do
      let gw = mkGridWorld
          (gw', reward, done) = stepEnv gw Down
      gwAgentPos gw' `shouldBe` (1, 0)
      reward `shouldBe` (-1.0)
      done `shouldBe` False

    it "moves agent right correctly" $ do
      let gw = mkGridWorld
          (gw', _, _) = stepEnv gw MoveRight
      gwAgentPos gw' `shouldBe` (0, 1)

    it "prevents moving past top wall" $ do
      let gw = mkGridWorld
          (gw', reward, _) = stepEnv gw Up
      gwAgentPos gw' `shouldBe` (0, 0)
      reward `shouldBe` (-2.0)

    it "prevents moving past left wall" $ do
      let gw = mkGridWorld
          (gw', reward, _) = stepEnv gw MoveLeft
      gwAgentPos gw' `shouldBe` (0, 0)
      reward `shouldBe` (-2.0)

    it "gives negative reward for hitting obstacle" $ do
      let gw = mkGridWorld { gwAgentPos = (0, 2) }
          (gw', reward, done) = stepEnv gw MoveRight
      gwAgentPos gw' `shouldBe` (0, 3)
      reward `shouldBe` (-100.0)
      done `shouldBe` True

    it "stateToVector produces correct size" $ do
      let gw = mkGridWorld
          sv = stateToVector gw
      size sv `shouldBe` stateSize

    it "stateToVector values are in reasonable range" $ do
      let gw = mkGridWorld
          sv = stateToVector gw
          vals = toList sv
      all (\v -> v >= -2.0 && v <= 2.0) vals `shouldBe` True

  describe "NeuralNetwork" $ do
    it "initializes network with correct bias dimensions" $ do
      let gen = mkStdGen 42
          (net, _) = initNetwork gen 12 64 4
      size (netBias1 net) `shouldBe` 64
      size (netBias2 net) `shouldBe` 4

    it "forward pass produces output of correct size" $ do
      let gen = mkStdGen 42
          (net, _) = initNetwork gen 12 64 4
          input' = fromList (replicate 12 0.5)
          (output', _, _) = forward net input'
      size output' `shouldBe` 4

    it "predict returns same as forward output" $ do
      let gen = mkStdGen 42
          (net, _) = initNetwork gen 12 64 4
          input' = fromList (replicate 12 0.5)
          predicted = predict net input'
          (output', _, _) = forward net input'
      toList predicted `shouldBe` toList output'

    it "networkLoss is zero for identical vectors" $ do
      let v = fromList [1.0, 2.0, 3.0, 4.0]
      networkLoss v v `shouldBe` 0.0

    it "networkLoss is positive for different vectors" $ do
      let v1 = fromList [1.0, 2.0, 3.0, 4.0]
          v2 = fromList [1.1, 2.1, 3.1, 4.1]
      networkLoss v1 v2 `shouldSatisfy` (> 0)

  describe "ReplayBuffer" $ do
    it "creates empty buffer" $ do
      let buf = emptyBuffer 100
      bufferSize buf `shouldBe` 0

    it "adds experience to buffer" $ do
      let buf = emptyBuffer 100
          exp' = Experience
            { expState = fromList (replicate 12 0.0)
            , expAction = Up
            , expReward = 1.0
            , expNextState = fromList (replicate 12 0.0)
            , expDone = False
            }
          buf' = addExperience exp' buf
      bufferSize buf' `shouldBe` 1

    it "respects maximum buffer size" $ do
      let buf = emptyBuffer 3
          mkExp r = Experience
            { expState = fromList (replicate 12 0.0)
            , expAction = Up
            , expReward = r
            , expNextState = fromList (replicate 12 0.0)
            , expDone = False
            }
          buf' = foldl (\b r -> addExperience (mkExp r) b) buf [1..5]
      bufferSize buf' `shouldBe` 3

    it "samples correct batch size" $ do
      let buf = emptyBuffer 100
          mkExp r = Experience
            { expState = fromList (replicate 12 0.0)
            , expAction = Up
            , expReward = r
            , expNextState = fromList (replicate 12 0.0)
            , expDone = False
            }
          buf' = foldl (\b r -> addExperience (mkExp r) b) buf [1..10]
          gen = mkStdGen 42
          (batch, _) = sampleBatch gen 5 buf'
      length batch `shouldBe` 5

  describe "Agent" $ do
    it "defaultAgentConfig has epsilon 1.0" $
      acEpsilon defaultAgentConfig `shouldBe` 1.0

    it "decayEpsilon reduces epsilon" $ do
      let cfg = defaultAgentConfig
          cfg' = decayEpsilon cfg
      acEpsilon cfg' `shouldSatisfy` (< acEpsilon cfg)

    it "epsilon does not go below minimum" $ do
      let cfg = defaultAgentConfig { acEpsilon = 0.01 }
          cfg' = decayEpsilon cfg
      acEpsilon cfg' `shouldSatisfy` (>= acEpsilonMin cfg)

    it "selectAction returns valid action" $ do
      let gen = mkStdGen 42
          (net, gen') = initNetwork gen 12 64 4
          state' = fromList (replicate 12 0.5)
          (action, _) = selectAction gen' 0.0 net state'
      action `shouldSatisfy` (`elem` allActions)
