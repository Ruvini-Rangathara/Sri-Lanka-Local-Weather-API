# NASA POWER Weather Forecast API

A Flask-based weather forecasting API that provides temperature and precipitation predictions for Sri Lanka using NASA POWER data and Prophet forecasting models.

## 🌟 Features

- **Weather Forecasting**: 30-day temperature (T2M) and precipitation (PRECIP) forecasts
- **Historical Data**: Access to historical weather data from NASA POWER
- **Spatial Interpolation**: Bilinear interpolation for locations between grid points
- **Smart Caching**: Automatic caching of NASA POWER data and trained models
- **High Performance**: Parallel pre-warming of models for faster response times
- **RESTful API**: Simple HTTP endpoints for easy integration

## 🚀 Technology Stack

- **Backend**: Flask (Python)
- **Data Source**: NASA POWER API
- **Forecasting**: Facebook Prophet
- **Caching**: Local file-based caching
- **Interpolation**: Custom bilinear interpolation algorithm
- **Parallel Processing**: ThreadPoolExecutor for model pre-warming

## 📊 Data Sources

- **NASA POWER**: Primary data source for historical weather data
- **Parameters**: T2M (temperature at 2 meters) and PRECIP (precipitation)
- **Grid Resolution**: ~0.5° (approximately 55km at equator)
- **Temporal Range**: Daily data from 2018-01-01 to yesterday

## 📁 Project Structure

```
.
├── data/
│   ├── cache/          # Cached NASA POWER data
│   └── models/         # Trained Prophet models
│       ├── T2M/        # Temperature models
│       └── PRECIP/     # Precipitation models
├── app.py              # Main Flask application
└── requirements.txt    # Python dependencies
```

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ruvini-Rangathara/Sri-Lanka-Local-Weather-API.git
   cd weather-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## 🔑 Environment Variables

Configure the application using these environment variables:

```env
# Regional settings
SL_LAT_MIN=5.8          # Southern boundary of Sri Lanka
SL_LAT_MAX=10.1         # Northern boundary
SL_LON_MIN=79.5         # Western boundary
SL_LON_MAX=82.1         # Eastern boundary
GRID_STEP=0.5           # Grid resolution in degrees

# Temporal settings
START_DATE=2018-01-01   # Start date for historical data
END_DATE=auto           # End date (defaults to yesterday)

# NASA POWER settings
COMMUNITY=AG            # NASA POWER community (AG for agriculture)
POWER_PARAMS=T2M,PRECTOTCORR  # Parameters to fetch

# Performance settings
PREWARM_WORKERS=4       # Number of parallel workers for pre-warming
PREWARM_SLEEP=0.05      # Sleep between requests to avoid rate limiting
MAX_HORIZON=30          # Forecast horizon in days
```

## 📋 API Endpoints

### Health Check
```
GET /health
```
Returns API status and configuration.

### Forecast Endpoint
```
GET /forecast?lat=6.9271&lon=79.8612&vars=T2M,PRECIP&interp=1
```
Parameters:
- `lat`, `lon`: Coordinates (required)
- `vars`: Comma-separated variables (T2M, PRECIP)
- `interp`: Set to 1 for interpolation between grid points
- `date`: Optional specific date within forecast period

### Historical Data Endpoint
```
GET /history?lat=6.9271&lon=79.8612&date=2023-05-15&vars=T2M,PRECIP&interp=1
```
Parameters:
- `lat`, `lon`: Coordinates (required)
- `date`: Date in YYYY-MM-DD format (required)
- `vars`: Comma-separated variables (T2M, PRECIP)
- `interp`: Set to 1 for interpolation between grid points

## 🎯 Usage Examples

### Get 30-day forecast for Colombo
```bash
curl "http://localhost:8000/forecast?lat=6.9271&lon=79.8612"
```

### Get historical weather for a specific date
```bash
curl "http://localhost:8000/history?lat=6.9271&lon=79.8612&date=2023-05-15"
```

### Get forecast with interpolation for precise location
```bash
curl "http://localhost:8000/forecast?lat=6.9271&lon=79.8612&interp=1"
```

## 🔄 Model Training

The application automatically:
1. Fetches historical data from NASA POWER API
2. Trains Prophet models for each grid point
3. Validates models on a holdout set
4. Caches models for future use
5. Provides bilinear interpolation for locations between grid points

## 📈 Response Format

### Forecast Response
```json
{
  "code": 200,
  "message": "OK",
  "data": {
    "location": {
      "requested": {"lat": 6.9271, "lon": 79.8612},
      "mode": "interpolated",
      "neighbors": [
        {"lat": 6.5, "lon": 79.5},
        {"lat": 6.5, "lon": 80.0},
        {"lat": 7.0, "lon": 79.5},
        {"lat": 7.0, "lon": 80.0}
      ]
    },
    "forecast_window": {
      "start": "2023-06-01",
      "end": "2023-06-30"
    },
    "daily": [
      {
        "date": "2023-06-01",
        "t2m": 27.5,
        "t2m_lo": 26.2,
        "t2m_hi": 28.8,
        "precip": 2.1,
        "precip_lo": 0.5,
        "precip_hi": 4.3
      }
    ]
  }
}
```

## 🚦 Performance Optimization

### Pre-warming Models
To improve first-response performance, pre-warm models at startup:
```bash
PREWARM=1 python app.py
```

This will train models for all grid points in Sri Lanka before starting the server.

### Caching Strategy
- NASA POWER data is cached locally to avoid repeated API calls
- Trained models are cached for fast forecasting
- Automatic cache validation ensures data freshness

## 🤝 Error Handling

The API returns appropriate HTTP status codes:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters or missing data

Error responses include descriptive messages to help with debugging.

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

**NASA POWER Weather API** - Accurate weather forecasting for Sri Lanka using NASA data and machine learning!