import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool, cpu_count
from fpdf import FPDF
import uuid
import os

# Set up logging to show progress of the simulations
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load configuration from CSV file
def load_config(filename='config.csv'):
    df = pd.read_csv(filename)
    config = {
        'items': df.to_dict('records'),
        'simulations': 10000  # Number of simulations can be set directly in the script or read from another source
    }
    return config

# Function to generate random price based on distribution type
def generate_price(price_range, distribution):
    low, high = price_range
    
    if distribution == 'uniform':
        return np.random.uniform(low, high)
    elif distribution == 'normal':
        mean = (low + high) / 2
        std_dev = (high - low) / 4  # Assume 95% of values are within the range
        return np.random.normal(mean, std_dev)
    elif distribution == 'left-skewed':
        alpha = 5  # Adjust alpha for skewness
        skewed_dist = stats.beta(a=alpha, b=1)
        return low + (high - low) * skewed_dist.rvs()
    elif distribution == 'right-skewed':
        alpha = 5  # Adjust alpha for skewness
        skewed_dist = stats.beta(a=1, b=alpha)
        return low + (high - low) * skewed_dist.rvs()
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

# Function to run a single simulation
def run_single_simulation(config_items):
    total_cost = 0
    for item in config_items:
        # Determine if the item should be included in the simulation
        if item['compulsory'] or (not item['compulsory'] and np.random.rand() < item['inclusion_percentage'] / 100.0):
            item_total_cost = 0
            for _ in range(item['quantity']):
                price = generate_price((item['price_range_low'], item['price_range_high']), item['distribution'])
                item_total_cost += price
            total_cost += item_total_cost
    return total_cost

# Function to run Monte Carlo simulations with multiprocessing
def run_simulations_multiprocessing(config):
    num_simulations = config['simulations']
    config_items = config['items']
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_simulation, [config_items] * num_simulations)
        
    return results

# Function to plot skewed distributions
def plot_skewed_distribution(alpha, skew_type):
    x = np.linspace(0, 1, 1000)
    
    if skew_type == 'left':
        y = stats.beta(a=alpha, b=1).pdf(x)
        title = 'Left-Skewed Distribution (Beta Distribution)'
    elif skew_type == 'right':
        y = stats.beta(a=1, b=alpha).pdf(x)
        title = 'Right-Skewed Distribution (Beta Distribution)'
    else:
        raise ValueError("skew_type must be 'left' or 'right'")
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'Alpha = {alpha}')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot the cost distribution from simulations
def plot_cost_distribution(total_costs, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(total_costs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Costs from Monte Carlo Simulations')
    plt.xlabel('Total Cost ($)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)  # Save the plot to a file
    plt.close()  # Close the plot to free memory
    print(f"Plot saved as {output_path}")

# Function to generate a PDF report
def generate_pdf_report(config, results, output_pdf_path, graph_path):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Monte Carlo Simulation Report', 0, 1, 'C')

    # Config table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Configuration Settings', 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(30, 10, 'Item', 1)
    pdf.cell(20, 10, 'Quantity', 1)
    pdf.cell(40, 10, 'Price Range', 1)
    pdf.cell(30, 10, 'Distribution', 1)
    pdf.cell(30, 10, 'Compulsory', 1)
    pdf.cell(40, 10, 'Inclusion %', 1)
    pdf.ln()

    for item in config['items']:
        pdf.cell(30, 10, item['name'], 1)
        pdf.cell(20, 10, str(item['quantity']), 1)
        pdf.cell(40, 10, f"{item['price_range_low']} - {item['price_range_high']}", 1)
        pdf.cell(30, 10, item['distribution'], 1)
        pdf.cell(30, 10, str(item['compulsory']), 1)
        pdf.cell(40, 10, str(item.get('inclusion_percentage', 'N/A')), 1)
        pdf.ln()

    # Simulation results
    mean_cost = np.mean(results)
    median_cost = np.median(results)
    percentile_90_cost = np.percentile(results, 90)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Simulation Results', 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Estimated Mean Cost: ${mean_cost:.2f}", 0, 1)
    pdf.cell(0, 10, f"Estimated Median Cost: ${median_cost:.2f}", 0, 1)
    pdf.cell(0, 10, f"Estimated 90th Percentile Cost: ${percentile_90_cost:.2f}", 0, 1)

    # Add the graph to the PDF
    pdf.image(graph_path, x = None, y = None, w = 180)
    
    pdf.output(output_pdf_path)
    print(f"PDF report saved as {output_pdf_path}")

# Main function to run the simulation and display results
def main():
    config = load_config('config.csv')
    total_costs = run_simulations_multiprocessing(config)
    
    # Generate a unique filename for the PDF and graph
    unique_id = uuid.uuid4()
    output_graph_path = f"/app/output/cost_distribution_{unique_id}.png"
    output_pdf_path = f"/app/output/monte_carlo_report_{unique_id}.pdf"
    
    # Plot the distribution of total costs from simulations
    plot_cost_distribution(total_costs, output_graph_path)

    # Generate PDF report
    generate_pdf_report(config, total_costs, output_pdf_path, output_graph_path)

if __name__ == "__main__":
    main()
