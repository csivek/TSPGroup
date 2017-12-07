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
            this.time_limit = time*1000;                        // time is entered in the GUI in seconds, but timer wants it in milliseconds

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
            routePenStyle = new Pen(Color.Blue,1);
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
            this.time_limit = timelimit*1000;                                   //convert seconds to milliseconds
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
            float width  = g.VisibleClipBounds.Width-45F;
            float height = g.VisibleClipBounds.Height-45F;
            Font labelFont = new Font("Arial", 10);

            // Draw lines
            if (bssf != null)
            {
                // make a list of points. 
                Point[] ps = new Point[bssf.Route.Count];
                int index = 0;
                foreach (City c in bssf.Route)
                {
                    if (index < bssf.Route.Count -1)
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[index+1]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
                    else 
                        g.DrawString(" " + index +"("+c.costToGetTo(bssf.Route[0]as City)+")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
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
        public double costOfBssf ()
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
            int i, swap, temp, count=0;
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
        int totalStates;

        /// <summary>
        /// performs a Branch and Bound search of the state space of partial tours
        /// stops when time limit expires and uses BSSF as solution
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] bBSolveProblem()
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();

            string[] results = new string[3];
            heapProblems = new bBSubProblem[500];
            heapSize = 1;
            maxHeapSize = 1;
            int numSolutions = 0;

            InitializeRootProblem();

            results = defaultSolveProblem(); //set bssf to cheap solution
            double bssfCost = costOfBssf();
            int rejectedBranches = 0;
            totalStates = 1;

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
                        if (toExplore.travelCosts[prev,i] < double.PositiveInfinity)
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
                                else if (i != 0)
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

        private void AddToHeap(bBSubProblem newProb)
        {
            totalStates++;
            if (heapSize == heapProblems.Length - 1) doubleSize();

            int pos = ++heapSize;
            if (maxHeapSize < heapSize) { maxHeapSize = heapSize; }

            heapProblems[pos] = newProb;
            DecreaseKey(pos);

        }

        private void doubleSize()
        {
            bBSubProblem[] placeholder = new bBSubProblem[heapProblems.Length * 2];
            heapProblems.CopyTo(placeholder, 0);
            heapProblems = placeholder;
        }

        private bBSubProblem DeleteMin()
        {
            bBSubProblem prob = heapProblems[1];
            heapProblems[1] = heapProblems[heapSize];
            heapProblems[heapSize] = null;
            heapSize--;
            InceaseKey(1);
            return prob;
            
        }

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

        private double ComputeValue(int loc)
        {
            if (loc >= heapProblems.Length || heapProblems[loc] == null)
            { return double.PositiveInfinity; }
            return heapProblems[loc].lowestBound - 100 * Cities.Length * heapProblems[loc].path.Count;
        }

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

        private void RemoveTravelCosts(bBSubProblem prob, int row, int col)
        {
            for (int i = 0; i < Cities.Length; i++)
            {
                prob.travelCosts[row, i] = double.PositiveInfinity;
                prob.travelCosts[i, col] = double.PositiveInfinity;
            }
            prob.travelCosts[col, row] = double.PositiveInfinity;
        }

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

        /// <summary>
        /// finds the greedy tour starting from each city and keeps the best (valid) one
        /// </summary>
        /// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
        public string[] greedySolveProblem()
        {
            string[] results = new string[3];

            // TODO: Add your implementation for a greedy solver here.

            results[COST] = "not implemented";    // load results into array here, replacing these dummy values
            results[TIME] = "-1";
            results[COUNT] = "-1";

            return results;
        }

        public string[] fancySolveProblem()
        {
            string[] results = new string[3];

            // TODO: Add your implementation for your advanced solver here.

            results[COST] = "not implemented";    // load results into array here, replacing these dummy values
            results[TIME] = "-1";
            results[COUNT] = "-1";

            return results;
        }
        #endregion
    }

}
