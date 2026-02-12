# Movie Recommendation Service README
This README provides a step-by-step guide for installing and setting up the Movie Recommendation Service along with Prometheus and Grafana for monitoring the service's performance through a metric dashboard.
**The Project report is in Assignment 6 folder**
## Prerequisites
- Anaconda (for managing Python environments)
- Prometheus (for monitoring service metrics)
- Grafana (for visualizing the metrics in a dashboard)

## Part 1: Installing and Setting Up the Movie Recommendation Service
### Step 1: Set Up Conda Environment
Open your terminal/command prompt and set up a conda environment for the service.

```
conda env create -f playground_environment.yml
```

### Step 2: Activate the Environment
Activate the environment you just created.

```
conda activate movie_recommendation
```

### Step 3: Start the Movie Recommendation Service
Run the dashboard by executing the following command in your terminal:

```
python dashboard.py
```

### Step 4: Start all services

Once startet a microservice dashboard will show up that looks like this:
![Microservice Dashboard Stopped](Project/images/microservice_dashboard_stopped.PNG)

Here you will need to start all services.
This may take some time for all the services to fully start and load.

Once loaded it should look like this:

![Microservce Dasboard Running](Project/images/microservice_dashboard_running.PNG)

### Step 5: Open the Movie Recommendation Dashboard
Once the service is running, open the `index.html` file located in the project folder to access the movie recommendation service in your browser.


## Part 2: Installing Prometheus and Grafana for Metric Monitoring
### Step 1: Install Prometheus
Download Prometheus from the official website.

Extract the downloaded files and navigate to the Prometheus folder.

### Step 2: Configure Prometheus
Open the `prometheus.yml` file in the Prometheus folder and replace its content with the `prometheus.yml` file provided in the GitLab repository.

### Step 3: Start Prometheus
Start Prometheus using the following command from the Prometheus folder:

```
prometheus --config.file=prometheus.yml
```

This will start Prometheus and begin collecting metrics for your service.

### Step 4: Install and Configure Grafana
Download Grafana from the official website.

After installation, start `grafana-server.exe` in `\GrafanaLabs\grafana\bin`

### Step 5: Open Grafana
Access Grafana by opening the following URL in your browser:

```
http://127.0.0.1:3000
```

Log in using the default credentials:

- Username: `admin`
- Password: `admin`

### Step 6: Add Prometheus data to Grafana

Go to Add Data Source and Select Prometheus as a new Data Source.

From there Grafana will ask you for an IP, fill in 

```
http://127.0.0.1:9090
```

### Step 7: Import Dashboard in Grafana

Go to Dashboards:

- Click on "New" in the Dashboards menu.
- Then click on "Import".
- Click "Upload JSON file" and select the grafana_dashboard.json file provided in the GitLab repository.
- Click "Import".
- Select Data sources as Prometheus

Your metric dashboard should now be loaded and displaying the service's metrics.
![Grafana Dashboard](Project/images/grafana-dashboard.PNG)
