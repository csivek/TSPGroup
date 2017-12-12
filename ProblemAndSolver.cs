using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;


namespace TSP
{

    class ProblemAndSolver
    {

        private class TSPSolution
        {
            /// <summary>
            /// we use the representation [cityB,cityA,cityC] 
            /// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
            /// and the edge from cityC to cityB is the final edge in the path.  
            /// You are, of course, free to use a different representation if it would be more convenient or efficient 
            /// for your data structure(s) and search algorithm. 
            /// </summary>
            public ArrayList
                Route;


            /// <summary>
            /// constructor
            /// </summary>
            /// <param name="iroute">a (hopefully) valid tour</param>
            public TSPSolution(ArrayList iroute)
            {
                Route = new ArrayList(iroute);
            }

            /// <summary>
            /// Compute the cost of the current route.  
            /// Note: This does not check that the route is complete.
            /// It assumes that the route passes from the last city back to the first city. 
            /// </summary>
            /// <returns></returns>
            public double costOfRoute()
            {
                // go through each edge in the route and add up the cost. 
                int x;
                City here;
                double cost = 0D;

                for (x = 0; x < Route.Count - 1; x++)
                {
                    here = Route[x] as City;
                    cost += here.costToGetTo(Route[x + 1] as City);
                }

                // go from the last city to the first. 
                here = Route[Route.Count - 1] as City;
                cost += here.costToGetTo(Route[0] as City);
                return cost;
            }
        }

        #region Private members

        /// <summary>
        /// Default number of cities (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Problem Size text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int DEFAULT_SIZE = 25;


        public bool drawAnts = false;
        /// <summary>
        /// Default time limit (unused -- to set defaults, change the values in the GUI form)
        /// </summary>
        // (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
        // click on the Time text box, go to the Properties window (lower right corner), 
        // and change the "Text" value.)
        private const int TIME_LIMIT = 60;        //in seconds

        private const int CITY_ICON_SIZE = 5;


        // For normal and hard modes:
        // hard mode only
        private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

        /// <summary>
        /// the cities in the current problem.
        /// </summary>
        private City[] Cities;
        /// <summary>
        /// a route through the current problem, useful as a temporary variable. 
        /// </summary>
        private ArrayList Route;
        /// <summary>
        /// best solution so far. 
        /// </summary>
        private TSPSolution bssf;

        /// <summary>
        /// how to color various things. 
        /// </summary>
        private Brush cityBrushStartStyle;
        private Brush cityBrushStyle;
        private Pen routePenStyle;


        /// <summary>
        /// keep track of the seed value so that the same sequence of problems can be 
        /// regenerated next time the generator is run. 
        /// </summary>
        private int _seed;
        /// <summary>
        /// number of cities to include in a problem. 
        /// </summary>
        private int _size;

        /// <summary>
        /// Difficulty level
        /// </summary>
        private HardMode.Modes _mode;

        /// <summary>
        /// random number generator. 
        /// </summary>
        private Random rnd;

        /// <summary>
        /// time limit in milliseconds for state space search
        /// can be used by any solver method to truncate the search and return the BSSF
        /// </summary>
        private int time_limit;
        #endregion

        #region Public members

        /// <summary>
        /// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
        /// </summary>
        public const int COST = 0;
        public const int TIME = 1;
        public const int COUNT = 2;

        public int Size
        {
            get { return _size; }
        }

        public int Seed
        {
            get { return _seed; }
        }
        #endregion

        #region Constructors
        public ProblemAndSolver()
        {
            this._seed = 1;
            rnd = new Random(1);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed)
        {
            this._seed = seed;
            rnd = new Random(seed);
            this._size = DEFAULT_SIZE;
            this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }

        public ProblemAndSolver(int seed, int size)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = TIME_LIMIT * 1000;                        // TIME_LIMIT is in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        public ProblemAndSolver(int seed, int size, int time)
        {
            this._seed = seed;
            this._size = size;
            rnd = new Random(seed);
            this.time_limit = time * 1000;                        // time is entered in the GUI in seconds, but timer wants it in milliseconds

            this.resetData();
        }
        #endregion

        #region Private Methods

        /// <summary>
        /// Reset the problem instance.
        /// </summary>
        private void resetData()
        {

            Cities = new City[_size];
            Route = new ArrayList(_size);
            bssf = null;

            if (_mode == HardMode.Modes.Easy)
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
            }
            else // Medium and hard
            {
                for (int i = 0; i < _size; i++)
                    Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
            }

            HardMode mm = new HardMode(this._mode, this.rnd, Cities);
            if (_mode == HardMode.Modes.Hard)
            {
                int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
                mm.removePaths(edgesToRemove);
            }
            City.setModeManager(mm);

            cityBrushStyle = new SolidBrush(Color.Black);
            cityBrushStartStyle = new SolidBrush(Color.Red);
            routePenStyle = new Pen(Color.Blue, 1);
            routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// make a new problem with the given size.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode)
        {
            this._size = size;
            this._mode = mode;
            resetData();
        }

        /// <summary>
        /// make a new problem with the given size, now including timelimit paremeter that was added to form.
        /// </summary>
        /// <param name="size">number of cities</param>
        public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
        {
            this._size = size;
            this._mode = mode;
            this.time_limit = timelimit * 1000;                                   //convert seconds to milliseconds
            resetData();
        }

        /// <summary>
        /// return a copy of the cities in this problem. 
        /// </summary>
        /// <returns>array of cities</returns>
        public City[] GetCities()
        {
            City[] retCities = new City[Cities.Length];
            Array.Copy(Cities, retCities, Cities.Length);
            return retCities;
        }

        /// <summary>
        /// draw the cities in the problem.  if the bssf member is defined, then
        /// draw that too. 
        /// </summary>
        /// <param name="g">where to draw the stuff</param>
        public void Draw(Graphics g)
        {
            float width = g.VisibleClipBounds.Width - 45F;
            float height = g.VisibleClipBounds.Height - 45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count - 1)
                        g.DrawString(" " + index + "(" + c.costToGetTo(bssf.Route[index + 1] as City) + ")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else
                        g.DrawString(" " + index + "(" + c.costToGetTo(bssf.Route[0] as City) + ")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
                }

                if (ps.Length > 0)
                {
                    g.DrawLines(routePenStyle, ps);
                    g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
                }

                // draw the last line. 
                g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
            }


            if (drawAnts)
            {
                City fc;
                City tc;
                Pen antPen = new Pen(Color.Black, 1);
                for (int j = 0; j < Cities.Length; j++) //evaporate pheremones between nodes
                {
                    fc = Cities[j];
                    for (int i = 0; i < Cities.Length; i++) //evaporate pheremones between nodes
                    {
                        tc = Cities[i];
                        if (i != j && fc.costToGetTo(tc) < Double.PositiveInfinity)
                        {
                            int redAmnt = (int)(nodes[j].getProbability(i) * 255);
                            redAmnt = (redAmnt < 0) ? 0 : (redAmnt > 255) ? 255 : redAmnt;
                            if (redAmnt > 50)
                            {
                                antPen.Color = Color.FromArgb(255 - redAmnt, redAmnt, 0); //(int)(nodes[j].getProbability(i) * 255), 0, 0);
                                g.DrawLine(antPen, (int)(fc.X * width) + CITY_ICON_SIZE / 2, (int)(fc.Y * height) + CITY_ICON_SIZE / 2,
                                    (int)(tc.X * width) + CITY_ICON_SIZE / 2, (int)(tc.Y * height) + CITY_ICON_SIZE / 2);
                            }
                        }
                    }
                }
            }



            // Draw city dots
            foreach (City c in Cities)
            {
                g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
            }

        }

        /// <summary>
        ///  return the cost of the best solution so far. 
        /// </summary>
        /// <returns></returns>
        public double costOfBssf()
        {
            if (bssf != null)
                return (bssf.costOfRoute());
            else
                return -1D;
        }

        /// <summary>
        /// This is the entry point for the default solver
        /// which just finds a valid random tour 
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] defaultSolveProblem()
        {
            int i, swap, temp, count = 0;
            string[] results = new string[3];
            int[] perm = new int[Cities.Length];
            Route = new ArrayList();
            Random rnd = new Random();
            Stopwatch timer = new Stopwatch();

            timer.Start();

            do
            {
                for (i = 0; i < perm.Length; i++)                                 // create a random permutation template
                    perm[i] = i;
                for (i = 0; i < perm.Length; i++)
                {
                    swap = i;
                    while (swap == i)
                        swap = rnd.Next(0, Cities.Length);
                    temp = perm[i];
                    perm[i] = perm[swap];
                    perm[swap] = temp;
                }
                Route.Clear();
                for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
                {
                    Route.Add(Cities[perm[i]]);
                }
                bssf = new TSPSolution(Route);
                count++;
            } while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
            timer.Stop();

            results[COST] = costOfBssf().ToString();                          // load results array
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = count.ToString();

            return results;
        }

        //Represents a Branch and Bound SubProblem
        public class bBSubProblem
        {
            public double lowestBound;
            public List<int> path;
            public double[,] travelCosts;

            public bBSubProblem()
            {
                lowestBound = double.PositiveInfinity;
                path = null;
                travelCosts = null;
            }
        }

        bBSubProblem[] heapProblems;
        int heapSize;
        int maxHeapSize;
        int totalProblemsMade;

        /// <summary>
        /// performs a Branch and Bound search of the state space of partial tours
        /// stops when time limit expires and uses BSSF as solution
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] bBSolveProblem()
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            //Initialization
            string[] results = new string[3];
            heapProblems = new bBSubProblem[500];
            heapSize = 1;
            maxHeapSize = 1;
            int numSolutions = 0;

            InitializeRootProblem();

            results = greedySolveProblem(); //set bssf to cheap solution
            double bssfCost = costOfBssf();
            int rejectedBranches = 0;
            totalProblemsMade = 1;

            //While the heap is not empty and time hasn't run out
            while (heapProblems[1] != null && timer.ElapsedMilliseconds < this.time_limit)
            {
                bBSubProblem toExplore = DeleteMin();
                if (toExplore.lowestBound < costOfBssf())
                {
                    //remember where you came from last
                    int prev = toExplore.path[toExplore.path.Count - 1];
                    for (int i = 0; i < Cities.Length; i++)
                    {
                        //if the path is takeable 
                        if (toExplore.travelCosts[prev, i] < double.PositiveInfinity)
                        {
                            bBSubProblem newProb = new bBSubProblem();
                            newProb.lowestBound = toExplore.lowestBound;
                            newProb.path = new List<int>(toExplore.path);
                            newProb.lowestBound += toExplore.travelCosts[prev, i];

                            if (newProb.lowestBound < costOfBssf())
                            {
                                //If the path is back to the beginning city, we found a solution
                                if (i == 0 && newProb.path.Count == Cities.Length)
                                {
                                    numSolutions++;
                                    ArrayList cPath = new ArrayList();
                                    for (int j = 0; j < newProb.path.Count; j++)
                                    {
                                        cPath.Add(Cities[newProb.path[j]]);
                                    }
                                    bssf = new TSPSolution(cPath);
                                }
                                else if (i != 0) //If you are worthy to explore then let's explore
                                {
                                    newProb.travelCosts = (double[,])toExplore.travelCosts.Clone();
                                    newProb.path.Add(i);
                                    RemoveTravelCosts(newProb, prev, i);
                                    UpdateTravelCosts(newProb, false);
                                    UpdateTravelCosts(newProb, true);

                                    AddToHeap(newProb);
                                }
                            }
                            else
                            {
                                rejectedBranches++;
                            }
                        }
                    }
                }
                else { rejectedBranches++; }
            }


            results[COST] = costOfBssf().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = numSolutions.ToString();

            return results;
        }

        //Adds to the Heap of Sub-Problems
        private void AddToHeap(bBSubProblem newProb)
        {
            totalProblemsMade++;
            if (heapSize == heapProblems.Length - 1) doubleSize();

            int pos = ++heapSize;
            if (maxHeapSize < heapSize) { maxHeapSize = heapSize; }

            heapProblems[pos] = newProb;
            DecreaseKey(pos);

        }

        //Creates more space for Sub-Problems if it needs to
        private void doubleSize()
        {
            bBSubProblem[] placeholder = new bBSubProblem[heapProblems.Length * 2];
            heapProblems.CopyTo(placeholder, 0);
            heapProblems = placeholder;
        }

        //Returns the top of the Heap and Adjusts
        private bBSubProblem DeleteMin()
        {
            bBSubProblem prob = heapProblems[1];
            heapProblems[1] = heapProblems[heapSize];
            heapProblems[heapSize] = null;
            heapSize--;
            InceaseKey(1);
            return prob;

        }

        //Move problem up the heap if it is lower
        private void DecreaseKey(int loc)
        {
            bBSubProblem tempProb;

            double locVal = ComputeValue(loc);
            int parentLoc = (loc / 2) + (loc % 2);
            double parVal = ComputeValue(parentLoc);

            //While I'm not at the root and my parent is greater than me
            while (loc != 1 && locVal < parVal)
            {
                tempProb = heapProblems[parentLoc];

                //Swap positions with the parent
                heapProblems[parentLoc] = heapProblems[loc];
                heapProblems[loc] = tempProb;

                loc = parentLoc;
                parentLoc = (loc / 2) + (loc % 2);
                parVal = ComputeValue(parentLoc);
            }
        }

        //move problem down the heap if it is higher
        private void InceaseKey(int loc)
        {
            int leftChildLoc = loc * 2;
            int rightChildLoc = loc * 2 + 1;
            bBSubProblem tempProb;

            double locVal = ComputeValue(loc);
            double leftVal = ComputeValue(leftChildLoc);
            double rightVal = ComputeValue(rightChildLoc);

            //While my children exist and I am greater than my children
            while ((leftChildLoc < heapProblems.Length && locVal > leftVal)
                || (rightChildLoc < heapProblems.Length && locVal > rightVal))
            {
                tempProb = heapProblems[loc];

                //Swap with the child with lowest distance
                if (rightChildLoc >= heapProblems.Length || leftVal < rightVal)
                {
                    heapProblems[loc] = heapProblems[leftChildLoc];
                    heapProblems[leftChildLoc] = tempProb;
                    loc = leftChildLoc;
                }
                else
                {
                    heapProblems[loc] = heapProblems[rightChildLoc];
                    heapProblems[rightChildLoc] = tempProb;
                    loc = rightChildLoc;
                }

                leftChildLoc = loc * 2;
                rightChildLoc = loc * 2 + 1;
                leftVal = ComputeValue(leftChildLoc);
                rightVal = ComputeValue(rightChildLoc);
            }
        }

        //Returns the Value of the Sub-Problem
        private double ComputeValue(int loc)
        {
            if (loc >= heapProblems.Length || heapProblems[loc] == null)
            { return double.PositiveInfinity; }
            return heapProblems[loc].lowestBound - 100 * Cities.Length * heapProblems[loc].path.Count;
        }

        //Initialization for the first Problem
        private void InitializeRootProblem()
        {
            heapProblems[1] = new bBSubProblem();
            heapProblems[1].lowestBound = 0;
            heapProblems[1].path = new List<int>();
            heapProblems[1].travelCosts = new double[Cities.Length, Cities.Length];
            heapProblems[1].path.Add(0); //Add start City

            for (int i = 0; i < Cities.Length; i++)
            {
                for (int j = 0; j < Cities.Length; j++)
                {
                    if (i == j) { heapProblems[1].travelCosts[i, j] = double.PositiveInfinity; }
                    else { heapProblems[1].travelCosts[i, j] = Cities[i].costToGetTo(Cities[j]); }
                }
            }

            UpdateTravelCosts(heapProblems[1], false);
            UpdateTravelCosts(heapProblems[1], true);
        }

        //Given an edge (row and column) this code will remove all edges
        //in the row and column and it's symmetric edge (edges[col, row]) 
        private void RemoveTravelCosts(bBSubProblem prob, int row, int col)
        {
            for (int i = 0; i < Cities.Length; i++)
            {
                prob.travelCosts[row, i] = double.PositiveInfinity;
                prob.travelCosts[i, col] = double.PositiveInfinity;
            }
            prob.travelCosts[col, row] = double.PositiveInfinity;
        }

        //Simplifies the travelCost matrix and adds to the lower bound
        // of the Sub-Problem. One does one pass in the direction you tell it
        private void UpdateTravelCosts(bBSubProblem prob, bool vert)
        {

            for (int i = 0; i < Cities.Length; i++)
            {
                //Find the lowest cost from the city
                double lowest_cost = double.PositiveInfinity;
                for (int j = 0; j < Cities.Length; j++)
                {
                    if (lowest_cost > prob.travelCosts[vert ? j : i, vert ? i : j])
                        lowest_cost = prob.travelCosts[vert ? j : i, vert ? i : j];
                }

                if (lowest_cost < double.PositiveInfinity)
                {
                    //Set the values subtracting the lowest cost
                    for (int j = 0; j < Cities.Length; j++)
                    {
                        prob.travelCosts[vert ? j : i, vert ? i : j] -= lowest_cost;
                    }

                    //Add the lowest cost to lowest cost possible path
                    prob.lowestBound += lowest_cost;
                }
            }
        }

        /////////////////////////////////////////////////////////////////////////////////////////////
        // These additional solver methods will be implemented as part of the group project.
        ////////////////////////////////////////////////////////////////////////////////////////////

        private class Edge : IComparable
        {
            public int from;
            public int to;
            public double cost;

            public Edge(int fromCity, int toCity, double travelCost)
            {
                from = fromCity;
                to = toCity;
                cost = travelCost;
            }

            public int CompareTo(object obj)
            {

                return cost.CompareTo(((Edge)obj).cost);
            }
        }

        /// <summary>
        /// finds the greedy tour starting from each city and keeps the best (valid) one
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] greedySolveProblem()
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            string[] results = new string[3];

            Edge[] edges = new Edge[Cities.Length * (Cities.Length - 1)];
            int iter = 0;

            for (int i = 0; i < Cities.Length; i++)
            {
                for (int j = 0; j < Cities.Length; j++)
                {
                    if (i != j)
                    {
                        edges[iter] = new Edge(i, j, Cities[i].costToGetTo(Cities[j]));
                        iter++;
                    }
                }
            }

            Array.Sort(edges);
            int[] prevCity = new int[Cities.Length];
            int[] nextCity = new int[Cities.Length];
            int numPicks = 0;
            bool loop = false;

            for (int i = 0; i < Cities.Length; i++)
            {
                prevCity[i] = -1;
                nextCity[i] = -1;
            }

            for (int i = 0; i < edges.Length; i++)
            {
                // if the respective cites have not been filled yet
                if (prevCity[edges[i].to] == -1 && nextCity[edges[i].from] == -1)
                {
                    loop = false;
                    iter = edges[i].to;
                    while (nextCity[iter] != -1)
                    {
                        if (nextCity[iter] == edges[i].from)
                        {
                            loop = true;
                            break;
                        }
                        iter = nextCity[iter];
                    }

                    if (!loop)
                    {
                        numPicks++;
                        prevCity[edges[i].to] = edges[i].from;
                        nextCity[edges[i].from] = edges[i].to;
                    }
                    else if (numPicks == Cities.Length - 1)
                    {
                        prevCity[edges[i].to] = edges[i].from;
                        nextCity[edges[i].from] = edges[i].to;
                        break;
                    }
                }
            }

            Route.Clear();
            iter = 0;
            Route.Add(Cities[0]);
            while (nextCity[iter] != 0)
            {
                iter = nextCity[iter];
                Route.Add(Cities[iter]);
            }

            bssf = new TSPSolution(Route);

            results[COST] = costOfBssf().ToString();
            results[TIME] = timer.Elapsed.ToString();
            results[COUNT] = "1";

            return results;
        }


        private class ACONode
        {
            // tuning parameters
            const double evaporation_factor = .1; //pheremone evaporation/forget factor
            const double alpha = .01;    //alpha weight for pheremone
            const double beta = 5;    //beta weight for distance

            int city;
            int num_cities;
            double[] probs;


            Double[] pher; //array of pheremons to each city
            Double[] dist; //array of distances to each city

            public ACONode(int city, int num_cities, City[] Cities)
            {
                this.city = city;
                this.num_cities = num_cities;
                pher = new Double[num_cities];
                dist = new Double[num_cities];
                probs = new double[num_cities];
                for (int i = 0; i < num_cities; i++)
                {
                    dist[i] = Cities[city].costToGetTo(Cities[i]);
                    pher[i] = .5;
                }
            }

            public int NextCity(List<int> visited, int starting_node)
            {
                double sum = 0;
                
                int num_visited = visited.Count;



                if ((num_visited - 1) == num_cities)
                {
                    return starting_node;
                }


                for (int i = 0; i < num_cities; i++) // get likelihoods of visiting cities based on Pheremon and Distance
                {
                    if (!visited.Contains(i) && city != i && i != starting_node)
                    {
                        if (dist[i] != 0 && dist[i] < double.PositiveInfinity)
                        {
                            probs[i] = Math.Pow(pher[i], alpha) * Math.Pow(1 / dist[i], beta);

                        }
                        else
                        {
                            probs[i] = 0;

                        }
                        sum += probs[i];
                    }
                    else
                    {
                        probs[i] = 0;
                    }
                }

                for (int i = 0; i < num_cities; i++) //scale to probabilites
                {
                    probs[i] = probs[i] / sum;
                }


                int choice = 0;
                double cumulative = 0;
                Random rnd = new Random();
                double stop = rnd.NextDouble();

                for (int i = 0; i < num_cities; i++) //randomly select which city to visit next
                {
                    if (probs[i] != 0)
                    {
                        cumulative += probs[i];
                        if (cumulative > stop)
                        {
                            choice = i;
                            break;
                        }
                    }
                }
                if (visited.Contains(choice))
                {
                    for (int i = 0; i < num_cities; i++)
                    {
                        if (!visited.Contains(i) && dist[i] < double.PositiveInfinity)
                        {
                            return i;
                        }
                    }
                }
                return choice;
            }

            public void addpheremones(int next_city, double value_added = .1) //add pherenomes between nodes
            {
                pher[next_city] = pher[next_city] + value_added;
            }

            public void evaporatepheremones()
            {
                for (int i = 0; i < num_cities; i++)
                {
                    pher[i] = (1 - evaporation_factor) * pher[i];
                }
            }

            public double getProbability(int i)
            {
                return probs[i];
            }

        }


        private class ant
        {
            int current_node;
            int initial_node;
            List<int> visited;
            double cost;
            bool random;
            int num_cities;

            public ant(bool random, int num_cities)
            {
                visited = new List<int>();
                this.random = random;
                this.num_cities = num_cities;
                reset();
            }


            public void reset() //reset ant location and cost to new journey
            {
                if (random)
                {
                    Random rnd = new Random();
                    initial_node = rnd.Next(0, num_cities); // randomly choose starting node
                }
                else
                {
                    initial_node = 0;
                }
                current_node = initial_node;
                cost = 0;
                visited.Clear();
            }

            public List<int> getvisited()
            {
                return visited;
            }

            public int getcurrentnode()
            {
                return current_node;
            }
            public int getinitialnode()
            {
                return initial_node;
            }

            public void visit(int city, double added_cost) //update node and cost
            {
                visited.Add(city);
                cost += added_cost;
                current_node = city;
            }

            public double getcost()
            {
                return cost;
            }
        }

        private ACONode[] nodes;


        public string[] fancySolveProblem(mainform form)
        {

            drawAnts = true;
            string[] results = new string[3];


            const int batch_size = 10; //tuning parameters
            const bool random_city = true;
            const double pherenome_constant = 5;


            ant[] ants = new ant[batch_size];
            nodes = new ACONode[Cities.Length];


            for (int i = 0; i < batch_size; i++)
            {
                ants[i] = new ant(random_city, Cities.Length); //instantiate ants, O(a)
            }

            for (int i = 0; i < Cities.Length; i++)
            {
                nodes[i] = new ACONode(i, Cities.Length, Cities);   //instantiace cities O(n)
            }


            double best_cost = double.PositiveInfinity;
            int updates = 0;
            int current_city;
            int next_city;
            int initial_city;
            double cost;
            double total_cost;
            double pheremone_total;
            List<int> visited;

            //results = greedySolveProblem(); //get bssf from greed solution

            //double best_cost = double.Parse(results[COST]);


            Stopwatch timer = new Stopwatch();
            timer.Start();


            while (timer.ElapsedMilliseconds < time_limit)
            {

                for (int j = 0; j < Cities.Length; j++) // iterate through every city
                {
                    for (int i = 0; i < ants.Length; i++) // for every ant move to a new city
                    {
                        current_city = ants[i].getcurrentnode();
                        initial_city = ants[i].getinitialnode();
                        next_city = nodes[current_city].NextCity(ants[i].getvisited(), initial_city);
                        cost = Cities[current_city].costToGetTo(Cities[next_city]);
                        nodes[current_city].addpheremones(next_city);
                        ants[i].visit(next_city, cost);
                    }
                }

                for (int i = 0; i < ants.Length; i++) //after ants finished route, for each ant
                {
                    total_cost = ants[i].getcost(); //find total cost
                    visited = ants[i].getvisited();



                    if (total_cost < best_cost)  //update best solution if found
                    {
                        Route.Clear();

                        for (int j = 0; j < visited.Count; j++)
                        {
                            Route.Add(Cities[visited[j]]);
                        }
                        bssf = new TSPSolution(Route);
                        best_cost = total_cost;
                        updates++;
                    }

                    pheremone_total = (total_cost / best_cost) * pherenome_constant; //get pheremone based on success of route

                    for (int j = 0; j < visited.Count; j++) //traverse through path adding pheremone based on success of route
                    {
                        current_city = visited[j];
                        if (j == (visited.Count - 1))
                        {
                            next_city = visited[0];
                        }
                        else
                        {
                            next_city = visited[j + 1];
                        }
                        nodes[current_city].addpheremones(next_city, pheremone_total);
                    }

                    ants[i].reset();        // reset ant to beginning of route 
                }

                for (int j = 0; j < Cities.Length; j++) //evaporate pheremones between nodes
                {
                    nodes[j].evaporatepheremones();
                }

                form.Refresh();

            } //repeat


            timer.Stop();

            results[COST] = best_cost.ToString();    // load results into array here, replacing these dummy values
            results[TIME] = timer.Elapsed.ToString(); ;
            results[COUNT] = updates.ToString();

            drawAnts = false;

            return results;
        }

        #endregion
    }

}
