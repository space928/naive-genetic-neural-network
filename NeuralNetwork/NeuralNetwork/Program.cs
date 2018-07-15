using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace NeuralNetwork
{
    class Program
    {
        public List<List<float>> networkNodes = new List<List<float>>();
        public List<List<List<float>>> networkConnections = new List<List<List<float>>>();
        public List<List<List<float>>> tempMutations = new List<List<List<float>>>();
        public long iters = 0;
        public long gens = 0;
        public long successIters = 0;
        public bool shouldPause = false;
        public DateTime startTime;
        public float currAnneal = 0.0f;
        public Queue<float> runningAvg = new Queue<float>();

        public struct Parents
        {
            public List<List<List<float>>> parent1Connections;
            public float parent1Fitness;
            public List<List<List<float>>> parent2Connections;
            public float parent2Fitness;
            public Parents(List<List<List<float>>> parent1Connections, float parent1Fitness, List<List<List<float>>> parent2Connections, float parent2Fitness)
            {
                this.parent1Connections = parent1Connections;
                this.parent1Fitness = parent1Fitness;
                this.parent2Connections = parent2Connections;
                this.parent2Fitness = parent2Fitness;
            }
            public Parents(Parents source)
            {
                this.parent1Connections = new List<List<List<float>>>(source.parent1Connections);
                this.parent1Fitness = source.parent1Fitness;
                this.parent2Connections = new List<List<List<float>>>(source.parent2Connections);
                this.parent2Fitness = source.parent2Fitness;
            }
        }

        //TODO: Migrate these to a settings file
        public const bool highPrecision = false;
        public const int nodeLayers = 6;
        public const int nodesPL = 8;
        public const int fitnessAverageIters = 400;
        public const float mutationRate = 1.5f;
        public const int childrenPerGen = 100;
        public const int overrallFitIters = 2000;
        //public const float annealAmount = 0.8f;//Start High
        //public const long annealSettleIters = 50000;//After this many iterations the annealing will stop
        //public const int annealDelay = 1000;

        public const string alphabet = "abcdefghijklmnopqrstuvwxyz '";
        public string[] trainData = new string[] { "thomas", "apple" };
        public string[] badData = new string[] { "xihfv", "vq" };

        //Small functions
        private List<float> GenerateRandomWord(Random rnd)
        {
            List<float> outp = new List<float>();
            int wl = (int)GetRandomIntBetween(1, nodesPL, rnd);
            for (int i = 0; i < nodesPL; i++)
            {
                if (i < wl)
                {
                    outp.Add((float)Math.Floor(GetRandomIntBetween(0, 26, rnd)));
                }
                else
                {
                    outp.Add(-1);
                }
            }

            if (highPrecision)
            {
                string strVersion = "";
                for (int i = 0; i < nodesPL; i++)
                {
                    if (outp[i] == -1)
                        break;
                    strVersion += alphabet[(int)outp[i]];
                }

                //This might be dangerous
                if (trainData.Contains(strVersion))//This is far to slow
                    outp = GenerateRandomWord(rnd);
            }

            return outp;
        }

        public List<List<List<float>>> DeepCopyConnections(List<List<List<float>>> source)
        {
            List<List<List<float>>> outp = new List<List<List<float>>>();
            for (int x = 0; x < nodeLayers; x++)
            {
                outp.Add(new List<List<float>>());
                for (int y = 0; y < nodesPL; y++)
                {
                    outp[x].Add(new List<float>());
                    for (int z = 0; z < nodesPL; z++)
                    {
                        outp[x][y].Add(source[x][y][z]);
                    }
                }
            }
            return outp;
        }

        public float GetRandomIntBetween(int x, int y, Random rnd)
        {
            return (float)(rnd.NextDouble() * (y - x) + x);
        }

        public float Clamp01(float x)
        {
            if (x > 1)
                return 1;
            else if (x < 0)
                return 0;
            else
                return x;
        }

        public List<float> WordToFloats(string word)
        {
            //Console.WriteLine("Evaluating: " + word + "...");

            List<float> outp = new List<float>();
            foreach (char letter in word)
            {
                outp.Add((float)alphabet.IndexOf(letter));
            }
            for (int i = outp.Count - 1; i < nodesPL; i++)
            {
                outp.Add(-1);
            }

            return outp;
        }

        /// <summary>
        /// Program entry
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            Program myProgram = new Program();
            myProgram.Start();
        }

        /// <summary>
        /// Main logic and UI
        /// </summary>
        public void Start()
        {
            //Startup UI
            Console.Title = "Neural Network By Thomas";
            Console.WriteLine("### Neural Network By Thomas ###");
            Console.WriteLine();
            Console.WriteLine("Enter previous network path, leave bank to continue: ");
            string inPath = Console.ReadLine();
            if (inPath.Length > 0)
            {
                try
                {
                    string json = System.IO.File.ReadAllText(inPath);
                    networkConnections = JsonConvert.DeserializeObject<List<List<List<float>>>>(json);
                } catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                    return;
                }
            }
            Console.WriteLine("Enter training data path, leave bank to continue: ");
            inPath = Console.ReadLine();
            if (inPath.Length > 0)
            {
                try
                {
                    trainData = System.IO.File.ReadAllLines(inPath);
                } catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }

            Console.WriteLine("Setting up network...");
            InitialiseNetwork();
            Console.WriteLine("Network set up as " + nodesPL + " X " + nodeLayers + " nodes.");

            Console.CancelKeyPress += BreakFromTraining;
            startTime = DateTime.Now;

            TrainNetwork();

            //I know I'm doing this wrong
            InterruptMenu();
            //Console.WriteLine("Press any key to exit...");
            //Console.ReadKey();
        }

        public void TrainNetwork()
        {
            Console.WriteLine("Training network...");
            Console.WriteLine();

            Random rnd = new Random();
            Parents potentialParents = new Parents();
            Parents currentParents = new Parents();
            while (true)
            {
                if (shouldPause)
                    return;

                FillMutations(mutationRate);
                //float avgFitness = 0;
                bool lastSuccess = false;

                //Let's make some new genes
                List<List<List<float>>> currConnections = new List<List<List<float>>>(networkConnections);
                for (int x = 0; x < nodeLayers; x++)
                {
                    for (int y = 0; y < nodesPL; y++)
                    {
                        for (int z = 0; z < nodesPL; z++)
                        {
                            if (currentParents.parent1Connections != null)
                            {
                                //if (rnd.NextDouble() >= 0.5)
                                    currConnections[x][y][z] = currentParents.parent1Connections[x][y][z];
                                //else
                                //    currConnections[x][y][z] = currentParents.parent2Connections[x][y][z];
                            }
                            currConnections[x][y][z] += tempMutations[x][y][z];
                        }
                    }
                }

                float fitness = TestConnections(currConnections, rnd);
                runningAvg.Enqueue(fitness);

                //This selects the two best children
                if (fitness > potentialParents.parent1Fitness)
                {
                    if(fitness > potentialParents.parent2Fitness)
                    {
                        //Better than both
                        if(potentialParents.parent1Fitness>potentialParents.parent2Fitness)
                        {
                            potentialParents.parent2Connections = currConnections;
                            potentialParents.parent2Fitness = fitness;
                        } else
                        {
                            potentialParents.parent1Connections = DeepCopyConnections(currConnections);
                            potentialParents.parent1Fitness = fitness;
                        }
                    } else
                    {
                        //Better than one
                        potentialParents.parent1Connections = DeepCopyConnections(currConnections);
                        potentialParents.parent1Fitness = fitness;
                    }
                } else if(fitness > potentialParents.parent2Fitness)
                {
                    //Better than the other
                    potentialParents.parent2Connections = currConnections;
                    potentialParents.parent2Fitness = fitness;
                }

                //Genetic Selection
                if(iters % childrenPerGen == childrenPerGen-1)
                { 
                    currentParents = new Parents(potentialParents);
                    networkConnections = DeepCopyConnections(currentParents.parent1Connections);
                    //Retest the previous best
                    potentialParents.parent1Fitness = TestConnections(currentParents.parent1Connections, rnd);
                    gens++;
                }

                if (runningAvg.Count > overrallFitIters)
                    runningAvg.Dequeue();

                if (iters % 9 == 0)
                {
                    float overallFit = 0;
                    foreach (float i in runningAvg)
                    {
                        overallFit += i;
                    }
                    overallFit /= runningAvg.Count;

                    Console.Write("[" + (DateTime.Now-startTime).ToString(@"h\:mm\:ss") + "] Training... Current fitness: " + Math.Round(fitness * 100)
                        + "%; Overall fitness: " + Math.Round(overallFit * 100.0f)
                        + "% \n[" + (DateTime.Now - startTime).ToString(@"h\:mm\:ss") + "] Best fitness (gen): " + Math.Round(potentialParents.parent1Fitness * 100)
                        + "%; Iterations: " + iters
                        + "; Generations: " + gens);
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                }

                if (lastSuccess)
                    successIters++;

                iters++;
            }
        }

        public float TestConnections(List<List<List<float>>> currConnections, Random rnd)
        {
            float avgFitness = 0;
            for (int i = 0; i < fitnessAverageIters; i++)
            {
                //lastSuccess = false;
                if (rnd.NextDouble() >= 0.5)
                {
                    EvaluateNetwork(WordToFloats(trainData[(int)(rnd.NextDouble() * trainData.Length)]), currConnections);
                    avgFitness += (Clamp01(networkNodes.Last()[0]));
                }
                else
                {
                    List<float> rndWord = GenerateRandomWord(rnd);
                    EvaluateNetwork(rndWord, currConnections);
                    avgFitness += (1 - Clamp01(networkNodes.Last()[0]));
                }
            }

            avgFitness /= fitnessAverageIters;
            return avgFitness;
        }

        /// <summary>
        /// Small menu which appears when training is interrupted
        /// </summary>
        public void InterruptMenu()
        {
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("## Training Interrupted!");
            Console.WriteLine("1. Evaluate string in current state. \n2. Save current state. \n3. Reset weights. \n4. Anneal now. \n0. Continue training.");
            string inp = Console.ReadLine();
            switch (inp)
            {
                case "1":
                    Console.WriteLine("Enter string to evaluate: ");
                    string evstr = Console.ReadLine();
                    EvaluateNetwork(WordToFloats(evstr), networkConnections);
                    Console.WriteLine("Network outputted: " + networkNodes.Last()[0]);
                    Console.WriteLine("Press any key to continue...");
                    Console.ReadKey();
                    break;
                case "2":
                    Console.WriteLine("Save as: ");
                    string inpath = Console.ReadLine();
                    try
                    {
                        System.IO.File.WriteAllText(inpath, JsonConvert.SerializeObject(networkConnections));
                    } catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }
                    break;
                case "3":
                    InitialiseNetwork();
                    shouldPause = false;
                    TrainNetwork();
                    break;
                case "4":
                    Console.WriteLine("Anneal strength as a decimal: ");
                    string inpower = Console.ReadLine();
                    float newAnneal = 0;
                    float.TryParse(inpower, out newAnneal);
                    Console.WriteLine("Annealing network for 1 iteration!");
                    FillMutations(currAnneal);
                    for (int x = 0; x < nodeLayers; x++)
                    {
                        for (int y = 0; y < nodesPL; y++)
                        {
                            for (int z = 0; z < nodesPL; z++)
                            {
                                networkConnections[x][y][z] += tempMutations[x][y][z];
                            }
                        }
                    }
                    break;
                case "0":
                    shouldPause = false;
                    TrainNetwork();
                    break;
            }
            InterruptMenu();
        }

        private void BreakFromTraining(object sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;

            shouldPause = true;

            //InterruptMenu();
        }

        /// <summary>
        /// Reset all nodes and fill arrays with blank data
        /// </summary>
        public void InitialiseNetwork()
        {
            for(int y = 0; y < nodeLayers; y++)
            {
                networkNodes.Add(new List<float>());
                networkConnections.Add(new List<List<float>>());
                for (int x = 0; x < nodesPL; x++)
                {
                    networkNodes[y].Add(0);
                    networkConnections[y].Add(new List<float>());
                    for(int z = 0; z < nodesPL; z++)
                    {
                        networkConnections[y][x].Add(0);
                    }
                }
            }
        }

        /// <summary>
        /// Fill the mutations table
        /// </summary>
        /// <param name="mutAmount"></param>
        public void FillMutations(float mutAmount)
        {
            Random rand = new Random();
            tempMutations.Clear();
            for (int y = 0; y < nodeLayers; y++)
            {
                tempMutations.Add(new List<List<float>>());
                for (int x = 0; x < nodesPL; x++)
                {
                    tempMutations[y].Add(new List<float>());
                    for (int z = 0; z < nodesPL; z++)
                    {
                        tempMutations[y][x].Add(((float)rand.NextDouble()*2-1)*mutAmount);
                    }
                }
            }
        }

        /// <summary>
        /// Evaluate the network based on a particular list of input nodes
        /// </summary>
        /// <param name="inputNodes"></param>
        public void EvaluateNetwork(List<float> inputNodes, List<List<List<float>>> inputConnections)
        {
            networkNodes.Clear();
            for (int y = 0; y < nodeLayers; y++)
            {
                networkNodes.Add(new List<float>());
                for (int x = 0; x < nodesPL; x++)
                {
                    networkNodes[y].Add(0);
                }
            }

            for (int i = 0; i < nodesPL; i++)
            {
                networkNodes[0][i] = inputNodes[i];
                //Console.WriteLine("Node at x: " + 0 + ", y: " + i + " = " + networkNodes[0][i]);
            }

            for (int x = 1; x < nodeLayers; x++)
            {
                for(int y = 0; y < nodesPL; y++)
                {
                    for(int z = 0; z < nodesPL; z++)
                        networkNodes[x][y] += networkNodes[x-1][z]*(inputConnections[x][y][z]);
                    //Console.WriteLine("Node at x: " + x + ", y: " + y + " = " + networkNodes[x][y]);
                }
            }
        }
    }
}
