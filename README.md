# OpenET Irrigation Advisory Demo

This application demonstrates how to use OpenET data to provide irrigation recommendations for agricultural fields. The app allows users to select a field by drawing on an interactive map or uploading a boundary file, then retrieves evapotranspiration (ET) data from the OpenET API to calculate irrigation needs.

## Features

- Interactive map for field selection
- Support for uploading field boundaries (GeoJSON, Shapefile, KML)
- Retrieval of ET data from OpenET API
- Simple water-balance based irrigation scheduling
- Visualization of ET data through time-series charts
- Irrigation recommendations based on ET and rainfall data

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your OpenET API key:
   ```
   OPENET_API_KEY=your_api_key_here
   ```
   
## Usage

Run the Streamlit app:
```
streamlit run app.py
```

Then follow the steps in the application:
1. Select your field by drawing on the map or uploading a boundary file
2. Choose a date range for analysis
3. Retrieve ET data
4. View the results and irrigation recommendations

## Data Source

This application uses data from [OpenET](https://openetdata.org), a multi-model ensemble-based evapotranspiration data platform. OpenET combines satellite imagery and climate data to provide field-scale estimates of evapotranspiration.

## Disclaimer

The irrigation recommendations provided by this app are estimates based on ET data and should be used as a guide only. Actual irrigation needs may vary based on specific field conditions, irrigation system efficiency, and local weather patterns.

## License

MIT License

## Acknowledgments

This project uses the OpenET API for evapotranspiration data. OpenET is a collaboration between NASA, Desert Research Institute, and other partners.