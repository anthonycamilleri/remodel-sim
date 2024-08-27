### README for Monte Carlo Simulation for Remodeling Project Costs

---

#### **Overview**

This Python script performs a Monte Carlo simulation to estimate the costs of a remodeling project based on a set of configurable items and settings. It generates a PDF report with detailed results, including the probability of staying within a specified budget, and outputs a CSV file containing detailed data for each simulation run.

#### **Features**

- **Cost Simulation**: Simulates the costs of a remodeling project based on configurable items and settings.
- **Probability Calculation**: Calculates the probability of staying within a user-defined budget.
- **Detailed Reporting**: Generates a PDF report with a summary of results, including a histogram of total costs and detailed simulation breakdowns.
- **CSV Export**: Outputs a CSV file with detailed information for each simulation, including item inclusion and costs.
- **Configurable Inputs**: Utilizes two CSV configuration files for flexibility in setting up item-specific variables and global settings.

---

#### **Usage Instructions**

1. **Prepare Configuration Files**:

   - **`config_items.csv`**: This file should contain details about each item, such as quantity, price range, distribution, and whether the item is compulsory or optional.

     **Example:**

     ```csv
     name,quantity,price_range_low,price_range_high,distribution,compulsory,inclusion_percentage
     Paint,10,20,50,uniform,True,
     Tile,100,2,5,normal,False,50
     Cabinet,5,200,500,left-skewed,False,75
     #... Add more items as needed
     ```

   - **`config_settings.csv`**: This file should include global settings such as the number of simulations to run and the budget.

     **Example:**

     ```csv
     parameter,value
     number_of_simulations,10000
     budget,40000
     ```

2. **Run the Python Script**:

   Make sure the script, `monte_carlo_simulation.py`, is in the same directory as your configuration files. Open a terminal or command prompt, navigate to the directory containing the script, and run:

   ```bash
   python monte_carlo_simulation.py
   ```

3. **Check Outputs**:

   - **PDF Report**: The script generates a PDF report (`monte_carlo_report_<unique_id>.pdf`) containing a summary of the simulation results, the probability of staying within the budget, a histogram of total costs, and detailed simulation breakdowns.
   - **CSV File**: A CSV file (`simulation_details_<unique_id>.csv`) is generated containing detailed data for each simulation, including item inclusion and costs.

4. **Review the Results**:

   - Open the PDF report to review the summary and detailed breakdown of the simulation results.
   - The CSV file provides a comprehensive dataset for further analysis or record-keeping.

---

#### **Explanation of What the Script Does**

1. **Load Configurations**:
   - Reads item-specific data from `config_items.csv`.
   - Reads global settings from `config_settings.csv`.

2. **Perform Monte Carlo Simulations**:
   - Uses multiprocessing to run simulations efficiently.
   - For each simulation, randomly determines whether each optional item is included based on its inclusion percentage.
   - Generates random prices for items based on their specified distribution and rounds up to the nearest 10 EUR increment.

3. **Calculate and Analyze Results**:
   - Calculates the total cost for each simulation.
   - Computes the probability of staying within the specified budget based on simulation outcomes.

4. **Generate Reports**:
   - Produces a PDF report with a histogram of costs and a detailed summary of results.
   - Outputs a CSV file with data for each simulation run, including costs and item inclusion.

---

#### **Warnings and Considerations**

- **Assumptions in Simulation**:
  - The script assumes that the cost distributions and quantities provided in the configuration files accurately reflect potential costs. Misconfigurations or unrealistic input values could lead to misleading results.

- **Probability and Risk**:
  - The calculated probability of staying within the budget is based on the provided data and assumptions. It does not guarantee outcomes but provides a statistical likelihood based on the model.

- **Inflation and Market Variability**:
  - The script does not account for potential changes in market prices or inflation, which could affect actual remodeling costs. Regular updates to the input data are recommended to reflect current conditions.

- **Computational Resources**:
  - Running a large number of simulations can be computationally intensive. Ensure your system has sufficient resources, or reduce the number of simulations if necessary.

- **Interpreting Results**:
  - Use caution when interpreting the results. The Monte Carlo simulation provides a range of possible outcomes, but individual project circumstances may vary.

By following these instructions and considerations, you can effectively use this Monte Carlo simulation script to estimate remodeling project costs and assess the likelihood of staying within budget.