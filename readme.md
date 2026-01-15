🛰️ Multi‑Temporal Urban Growth Monitoring in the UAE (2018–2024)
📌 Project Overview
Urban expansion represents one of the most significant land surface transformations in arid regions. The United Arab Emirates (UAE), characterized by rapid economic development and population growth, has experienced accelerated urbanization over the last decade.

This capstone project develops and implements an end‑to‑end remote sensing workflow to monitor, map, and quantify urban growth across the UAE between 2018 and 2024. The methodology leverages Sentinel‑2 satellite imagery, Google Earth Engine (GEE), spectral index engineering, and supervised machine learning (Random Forest) to generate accurate, reproducible, and scalable urban land‑cover products.

The project delivers:

Annual cloud‑free land‑cover maps

Quantitative urban growth statistics

Accuracy‑validated classification results

An interactive dashboard for visualization and decision support

🎯 Project Objectives
The main objectives of this project are to:

Develop a robust preprocessing pipeline for generating annual dry‑season Sentinel‑2 composites optimized for arid environments.

Design and apply desert‑adapted spectral indices to reduce confusion between urban and bare soil surfaces.

Produce high‑accuracy land‑cover classifications using a Random Forest classifier.

Quantify urban expansion rates, spatial patterns, and temporal dynamics from 2018 to 2024.

Build an interactive dashboard to visualize results and facilitate stakeholder interpretation.

🌍 Study Area
Region: United Arab Emirates (national scale)

Geographic Extent:

Longitude: 51.5°E – 56.5°E

Latitude: 22.5°N – 26.5°N

Key Urban Centers: Abu Dhabi, Dubai, Sharjah, Al Ain

Environment: Arid and hyper‑arid desert landscapes with strong spectral similarity between urban surfaces and bare soil.

🗂️ Data Sources
Primary Data
Sentinel‑2 Surface Reflectance (Level‑2A)

Source: COPERNICUS/S2_SR

Spatial Resolution: 10 m – 20 m

Temporal Coverage: 2018–2024

Season: Dry season (October–April)

Ancillary Data
UAE administrative boundaries (USDOS/LSIB SIMPLE)

Road networks (for validation and spatial interpretation)

Population grids (WorldPop)

VIIRS night‑time lights (independent urban extent verification)

⚙️ Methodology
The workflow consists of eight integrated phases:

Data Acquisition & Preprocessing

Cloud masking using Sentinel‑2 Scene Classification Layer (SCL)

Median dry‑season compositing

Spectral Index Computation

Core indices: NDVI, NDBI, MNDWI, BUI

Comprehensive index suite for arid environments

Training Data Generation

Manual digitization of reference polygons (Urban, Vegetation, Bare Soil, Water)

Pixel‑based sampling strategy

Supervised Classification
result: ( images/manual.png)
Random Forest classifier

Input features: spectral bands + spectral indices

Accuracy Assessment

70/30 training–validation split

Confusion matrix, Overall Accuracy, User’s & Producer’s Accuracy, Kappa coefficient

Post‑Classification Analysis

Spatial smoothing (majority filter)

Area calculation and change detection

Dashboard Development

Interactive visualization of maps, indices, and statistics

Validation & Quality Assurance

Cross‑comparison with ancillary datasets

Temporal consistency checks

📊 Key Results
Overall Classification Accuracy (2018): 90.8%

Kappa Coefficient: 0.859 (excellent agreement)

Total Urban Growth (2018–2024): +4,394.96 km²

Percentage Growth: +61.1%

Average Annual Growth Rate: ~10.18% per year

Urban expansion accelerated notably after 2021, aligning with national development strategies and population growth trends.

🖥️ Interactive Dashboard
The project includes an interactive dashboard featuring:

Annual land‑cover maps (2018–2024)

Spectral index visualization (NDVI, NDBI, BUI, MNDWI)

Urban growth statistics and charts

Exportable results for further analysis
result:( images/dashboard2.png )

🚧 Challenges & Limitations
Spectral confusion between urban and bare soil in desert environments

Misclassification in mountainous and transition zones

Spatial resolution limitations of Sentinel‑2 (10 m)

Single annual composite per year limits intra‑annual analysis

🔮 Future Work
Integration of very high‑resolution imagery (PlanetScope, WorldView)

Adoption of Deep Learning models (CNNs, Transformers)

Full time‑series analysis using all available Sentinel‑2 observations

Incorporation of topographic and slope‑based corrections

👩‍💻 Team
Chaabouni Yessmine
Sahar Feki


📄 License & Academic Use
This project was developed as an academic capstone project.
All scripts and outputs are provided for research and educational purposes.


