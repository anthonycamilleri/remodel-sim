import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool, cpu_count, current_process
from fpdf import FPDF
import uuid
import os
import csv

# Set up logging to show progress of the simulations
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load configuration from CSV files
def load_configs(items_filename='config_items.csv', settings_filename='config_settings.csv'):
    items_df = pd.read_csv(items_filename)
    settings_df = pd.read_csv(settings_filename)

    # Parse items configuration
    config_items = items_df.to_dict('records')

    # Parse settings configuration
    config_settings = settings_df.set_index('parameter')['value'].to_dict()
    config_settings['number_of_simulations'] = int(config_settings['number_of_simulations'])
    config_settings['budget'] = float(config_settings['budget'])

    return config_items, config_settings

# Function to generate random price based on distribution type, rounded to nearest 10 EUR
def generate_price(price_range, distribution):
    low, high = price_range
    
    if distribution == 'uniform':
        price = np.random.uniform(low, high)
    elif distribution == 'normal':
        mean = (low + high) / 2
        std_dev = (high - low) / 4  # Assume 95% of values are within the range
        price = np.random.normal(mean, std_dev)
    elif distribution == 'left-skewed':
        alpha = 5  # Adjust alpha for skewness
        skewed_dist = stats.beta(a=alpha, b=1)
        price = low + (high - low) * skewed_dist.rvs()
    elif distribution == 'right-skewed':
        alpha = 5  # Adjust alpha for skewness
        skewed_dist = stats.beta(a=1, b=alpha)
        price = low + (high - low) * skewed_dist.rvs()
    else:
        raise ValueError(f"Unknown distribution type: {distribution}")
    
    return np.ceil(price / 10) * 10  # Round up to nearest 10 EUR increment

# Function to run a single simulation
def run_single_simulation(args):
    simulation_index, config_items = args
    total_cost = 0
    simulation_details = {'simulation_index': simulation_index + 1}

    logging.info(f"Running Simulation no. {simulation_index + 1} on Processor {current_process().name}")

    for item in config_items:
        item_name = item['name']
        item_included = False
        item_cost = 0

        # Determine if the item should be included in the simulation
        if item['compulsory'] or (not item['compulsory'] and np.random.rand() < item['inclusion_percentage'] / 100.0):
            item_included = True
            item_total_cost = 0
            for _ in range(item['quantity']):
                price = generate_price((item['price_range_low'], item['price_range_high']), item['distribution'])
                item_total_cost += price
            total_cost += item_total_cost
            item_cost = item_total_cost

        # Log whether the item was included and its cost
        simulation_details[item_name] = {'included': item_included, 'cost': item_cost}

    return total_cost, simulation_details

# Function to run Monte Carlo simulations with multiprocessing
def run_simulations_multiprocessing(config_items, num_simulations):
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(run_single_simulation, [(i, config_items) for i in range(num_simulations)])
    
    total_costs, simulation_details_list = zip(*results)
    return total_costs, simulation_details_list

# Function to plot the cost distribution from simulations
def plot_cost_distribution(total_costs, output_path):
    plt.figure(figsize=(10, 6))
    plt.hist(total_costs, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Costs from Monte Carlo Simulations')
    plt.xlabel('Total Cost (EUR)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)  # Save the plot to a file
    plt.close()  # Close the plot to free memory
    print(f"Plot saved as {output_path}")

# Function to plot a pie chart of cost breakdown
def plot_cost_breakdown(simulation_details_list, config_items, output_path):
    total_costs_per_item = {item['name']: 0 for item in config_items}

    # Aggregate costs across all simulations
    for details in simulation_details_list:
        for item in config_items:
            item_name = item['name']
            if details[item_name]['included']:
                total_costs_per_item[item_name] += details[item_name]['cost']

    # Prepare data for pie chart
    labels = total_costs_per_item.keys()
    sizes = total_costs_per_item.values()

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.title('Cost Breakdown by Item')
    plt.savefig(output_path)
    plt.close()
    print(f"Pie chart saved as {output_path}")

# Function to generate a PDF report using "EUR" instead of the Euro sign
def generate_pdf_report(config_settings, total_costs, simulation_details_list, output_graph_path, output_pie_chart_path, output_pdf_path):
    # Calculate the probability of staying within budget
    within_budget = sum(1 for cost in total_costs if cost <= config_settings['budget'])
    probability_within_budget = (within_budget / len(total_costs)) * 100

    # Calculate additional statistics
    mean_cost = np.mean(total_costs)
    median_cost = np.median(total_costs)
    std_dev_cost = np.std(total_costs)
    percentile_90_cost = np.percentile(total_costs, 90)

    # Calculate most impactful costs
    total_costs_per_item = {item_name: 0 for item_name in simulation_details_list[0].keys() if item_name != 'simulation_index'}
    for details in simulation_details_list:
        for item_name in total_costs_per_item.keys():
            if details[item_name]['included']:
                total_costs_per_item[item_name] += details[item_name]['cost']
    most_impactful_costs = sorted(total_costs_per_item.items(), key=lambda x: x[1], reverse=True)

    pdf = FPDF()
    pdf.add_page()
    
    # Use standard font that supports basic characters
    pdf.set_font('Arial', '', 12)

    # Title and Probability of Staying Within Budget
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Monte Carlo Simulation Report', 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f'Probability of Staying Within Budget (EUR {config_settings["budget"]:.2f}): {probability_within_budget:.2f}%', 0, 1)
    
    # Add the histogram to the PDF
    pdf.image(output_graph_path, x = None, y = None, w = 180)

    # Add the pie chart to the PDF
    pdf.image(output_pie_chart_path, x = None, y = None, w = 180)

    # Add some statistics to the PDF
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Key Statistics:', 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f'Average Total Cost: EUR {mean_cost:.2f}', 0, 1)
    pdf.cell(0, 10, f'Median Total Cost: EUR {median_cost:.2f}', 0, 1)
    pdf.cell(0, 10, f'Standard Deviation of Total Cost: EUR {std_dev_cost:.2f}', 0, 1)
    pdf.cell(0, 10, f'90th Percentile of Total Cost: EUR {percentile_90_cost:.2f}', 0, 1)

    # List the most impactful costs
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Most Impactful Costs:', 0, 1)
    pdf.set_font("Arial", '', 12)
    for item, cost in most_impactful_costs[:5]:  # Show top 5 most impactful costs
        pdf.cell(0, 10, f'{item}: EUR {cost:.2f}', 0, 1)

    pdf.output(output_pdf_path)
    print(f"PDF report saved as {output_pdf_path}")

# Function to generate a CSV report for all simulations
def generate_csv_report(config_items, simulation_details_list, output_csv_path):
    if not simulation_details_list:
        logging.warning("No simulation details available to write to CSV.")
        return

    # Prepare the fieldnames for the CSV file
    fieldnames = ['simulation_index'] + [f"{item['name']}_included" for item in config_items] + [f"{item['name']}_cost" for item in config_items]
    
    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate over each simulation's details to write them to the CSV
        for details in simulation_details_list:
            row = {'simulation_index': details['simulation_index']}
            for item_name, item_details in details.items():
                if item_name != 'simulation_index':
                    row[f"{item_name}_included"] = item_details['included']
                    row[f"{item_name}_cost"] = item_details['cost']
            writer.writerow(row)
    
    print(f"CSV report saved as {output_csv_path}")


# Main function to run the simulation and display results
def main():
    config_items, config_settings = load_configs('config_items.csv', 'config_settings.csv')
    total_costs, simulation_details_list = run_simulations_multiprocessing(config_items, config_settings['number_of_simulations'])
    
    # Generate a unique filename for the PDF, graph, and CSV
    unique_id = uuid.uuid4()
    output_graph_path = f"/app/output/cost_distribution_{unique_id}.png"
    output_pie_chart_path = f"/app/output/cost_breakdown_{unique_id}.png"
    output_pdf_path = f"/app/output/monte_carlo_report_{unique_id}.pdf"
    output_csv_path = f"/app/output/simulation_details_{unique_id}.csv"
    
    # Plot the distribution of total costs from simulations
    plot_cost_distribution(total_costs, output_graph_path)

    # Plot the pie chart of cost breakdown
    plot_cost_breakdown(simulation_details_list, config_items, output_pie_chart_path)

    # Generate PDF and CSV reports
    generate_pdf_report(config_settings, total_costs, simulation_details_list, output_graph_path, output_pie_chart_path, output_pdf_path)
    generate_csv_report(config_items, simulation_details_list, output_csv_path)

if __name__ == "__main__":
    main()
