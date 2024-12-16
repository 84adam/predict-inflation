# predict_inflation.py

import pandas as pd
import numpy as np
from copy import deepcopy
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Loads and preprocesses macroeconomic data."""
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path, parse_dates=['Date'], index_col='Date').asfreq('M')
            df = df.interpolate().dropna()
            if df.empty:
                raise ValueError("No data loaded or all data was removed after cleaning")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

class MacroFeatureEngineering:
    """Feature engineering pipeline: PCA, momentum, volatility, seasonality."""
    def __init__(self, commodity_pca_components=2, momentum_windows=[3, 6, 12],
                 momentum_features=['M2', 'Consumer_Spending', 'CPI']):
        self.commodity_pca_components = commodity_pca_components
        self.momentum_windows = momentum_windows
        self.momentum_features = momentum_features
        self.commodity_scaler = StandardScaler()
        self.commodity_pca = PCA(n_components=commodity_pca_components)

    def fit(self, df):
        commodity_cols = [c for c in df.columns if 'Commod' in c]
        if commodity_cols:
            commodity_data = self.commodity_scaler.fit_transform(df[commodity_cols])
            self.commodity_pca.fit(commodity_data)
        return self

    def transform(self, df):
        df = df.copy()
        
        # Commodity factor
        commodity_cols = [c for c in df.columns if 'Commod' in c]
        if commodity_cols:
            c_data = self.commodity_scaler.transform(df[commodity_cols])
            factors = self.commodity_pca.transform(c_data)
            for i in range(self.commodity_pca_components):
                df[f'Commodity_Factor_{i+1}'] = factors[:, i]

        # Momentum and growth rates
        for col in ['M2', 'Consumer_Spending', 'CPI']:
            if col in df.columns:
                df[f'{col}_YoY'] = df[col].pct_change(12)*100
                for w in self.momentum_windows:
                    df[f'{col}_Mom_{w}M'] = df[f'{col}_YoY'].rolling(w).mean()

        # Yield curve features
        if all(x in df.columns for x in ['10Y_Yield', '2Y_Yield']):
            df['Yield_Curve_Slope'] = df['10Y_Yield'] - df['2Y_Yield']
        if all(x in df.columns for x in ['10Y_Yield', '5Y_Yield', '2Y_Yield']):
            df['Yield_Curve_Curvature'] = (2*df['5Y_Yield'] - df['2Y_Yield'] - df['10Y_Yield'])

        # Financial conditions
        if all(x in df.columns for x in ['Credit_Spread', 'VIX', 'Ted_Spread']):
            fin_data = StandardScaler().fit_transform(df[['Credit_Spread', 'VIX', 'Ted_Spread']])
            df['Financial_Conditions'] = fin_data.mean(axis=1)

        # Volatility features
        for col in ['Commodity_Factor_1', 'M2_YoY', 'CPI_YoY']:
            if col in df.columns:
                df[f'{col}_Vol'] = df[col].rolling(6).std()

        # Seasonality
        if 'Core_PCE' in df.columns:
            stl = STL(df['Core_PCE'].dropna(), period=12, robust=True)
            res = stl.fit()
            df['Core_PCE_Trend'] = res.trend
            df['Core_PCE_Seasonal'] = res.seasonal
            df['Core_PCE_Residual'] = res.resid

        df = df.dropna()
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

class RegimeDetector:
    """Detect economic regimes using a Markov-switching model."""
    def __init__(self, k_regimes=2):
        self.k_regimes = k_regimes
        self.model = None
        self.results = None
        
    def fit(self, df):
        """Fit Markov switching model on Core PCE growth rates."""
        inflation_rate = df['Core_PCE'].pct_change(12) * 100
        inflation_rate = inflation_rate.dropna()
        if len(inflation_rate) < 20:
            logging.warning("Not enough data points for regime detection.")
            return self
        
        try:
            self.model = sm.tsa.MarkovRegression(
                inflation_rate,
                k_regimes=self.k_regimes,
                trend='c',
                switching_variance=True
            )
            self.results = self.model.fit(disp=False)
        except Exception as e:
            logging.warning(f"Regime detection failed: {str(e)}")
        return self
        
    def predict_regime_probs(self, df):
        """Predict regime probabilities for given data."""
        if self.results is None:
            return pd.DataFrame(0.5, index=df.index, columns=[f'Regime_{i+1}_Prob' for i in range(self.k_regimes)])
        
        inflation_rate = df['Core_PCE'].pct_change(12)*100
        filtered_probs = self.results.filtered_marginal_probabilities
        probs_df = pd.DataFrame(
            filtered_probs,
            index=inflation_rate.index[-len(filtered_probs):],
            columns=[f'Regime_{i+1}_Prob' for i in range(self.k_regimes)]
        )
        probs_df = probs_df.reindex(df.index, method='ffill').fillna(0.5)
        return probs_df

class BayesianRegressionModel:
    """Bayesian linear regression with simplified uncertainty quantification."""
    def __init__(self):
        self.model = BayesianRidge(max_iter=300, tol=1e-3)
        self.X_scaler = StandardScaler()
        self.train_residuals = None
        
    def fit(self, X, y):
        X_scaled = self.X_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Estimate residual std dev from training data
        y_pred = self.model.predict(X_scaled)
        residuals = y - y_pred
        self.resid_std = np.std(residuals)
        # Ensure a minimum residual standard deviation to avoid zero intervals
        if self.resid_std < 0.01:
            self.resid_std = 0.01
        return self
        
    def predict(self, X):
        X_scaled = self.X_scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_intervals(self, X, alpha=0.05):
        # Using a normal approximation:
        # mean +/- z * resid_std
        # alpha=0.05 => 95% CI => z ~ 1.96
        z = 1.96
        X_scaled = self.X_scaler.transform(X)
        preds = self.model.predict(X_scaled)
        
        lower = preds - z * self.resid_std
        upper = preds + z * self.resid_std

        # Validate intervals: If somehow intervals are huge, cap them (safety check)
        interval_width = upper - lower
        max_width = 5.0  # Cap at 5 percentage points range
        too_wide = interval_width > max_width
        if any(too_wide):
            # Scale down intervals where too wide
            scale_factor = max_width / interval_width
            upper = preds + (upper - preds) * scale_factor
            lower = preds - (preds - lower) * scale_factor
        
        return preds, lower, upper

class GradientBoostingModel:
    """Gradient Boosting model for forecasting."""
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, X):
        return pd.Series(self.model.feature_importances_, index=X.columns)

class EnsembleForecaster:
    """Ensemble forecaster without VARMAX."""
    def __init__(self, models, regime_detector=None, window_size=12, target_col='Core_PCE'):
        self.models = models
        self.regime_detector = regime_detector
        self.window_size = window_size
        self.target_col = target_col
        self.weights = {name: 1/len(models) for name in models.keys()}
        self.performance_history = {name: [] for name in models.keys()}

    def predict(self, X_current, horizon):
        predictions = {}
        intervals = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X_current)
                if hasattr(model, 'predict_intervals'):
                    mean, lower, upper = model.predict_intervals(X_current)
                    # Use the mean prediction here
                    predictions[name] = float(mean[0])
                    intervals[name] = (float(lower[0]), float(upper[0]))
                else:
                    pred_val = float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)
                    predictions[name] = pred_val
                    # Default interval ±0.5
                    intervals[name] = (pred_val - 0.5, pred_val + 0.5)
            except Exception as e:
                logging.warning(f"Model {name} prediction failed: {str(e)}")
                # Fallback to a neutral prediction
                fallback = X_current[self.target_col].iloc[0] if self.target_col in X_current else 2.0
                predictions[name] = fallback
                intervals[name] = (fallback - 0.5, fallback + 0.5)

        if not predictions:
            logging.warning("No predictions obtained, using fallback=2.0")
            return 2.0, 1.5, 2.5

        # Weighted average
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            # fallback equal weights
            self.weights = {m: 1/len(self.models) for m in self.models.keys()}
            total_weight = 1.0

        mean_pred = sum(self.weights[m]*predictions[m] for m in predictions) / total_weight
        lower_pred = sum(self.weights[m]*intervals[m][0] for m in intervals) / total_weight
        upper_pred = sum(self.weights[m]*intervals[m][1] for m in intervals) / total_weight

        # Additional validation: Ensure intervals not inverted or too wide
        if lower_pred > upper_pred:
            lower_pred, upper_pred = upper_pred, lower_pred
        interval_width = upper_pred - lower_pred
        if interval_width > 5.0:
            # Scale down large intervals
            scale_factor = 5.0 / interval_width
            upper_pred = mean_pred + (upper_pred - mean_pred)*scale_factor
            lower_pred = mean_pred - (mean_pred - lower_pred)*scale_factor

        return mean_pred, lower_pred, upper_pred

def run_backtest(df, target_col='Core_PCE', horizons=[6, 9, 12], train_window=120):
    results = []
    
    # Create horizon-specific targets
    for h in horizons:
        df[f'Target_{h}M'] = df[target_col].shift(-h)
    df = df.dropna()

    feature_cols = [c for c in df.columns if not c.startswith('Target_') and c != target_col]

    tscv = TimeSeriesSplit(n_splits=5)

    for train_idx, test_idx in tscv.split(df):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        for h in horizons:
            X_train = train_data[feature_cols]
            y_train = train_data[f'Target_{h}M']

            # Just two models: GB and Bayesian
            gb = GradientBoostingModel().fit(X_train, y_train)
            bayes = BayesianRegressionModel().fit(X_train, y_train)

            models = {'gb': gb, 'bayes': bayes}

            ef = EnsembleForecaster(models=models, target_col=target_col)
            
            for i in range(len(test_data)):
                X_current = test_data.iloc[[i]][feature_cols]
                mean_pred, lower_pred, upper_pred = ef.predict(X_current, h)
                actual = test_data.iloc[i][f'Target_{h}M']

                results.append({
                    'Date': test_data.index[i],
                    'horizon': h,
                    'prediction': mean_pred,
                    'lower_ci': lower_pred,
                    'upper_ci': upper_pred,
                    'actual': actual,
                    'error': mean_pred - actual
                })
                
    return pd.DataFrame(results)

def create_inflation_signals(predictions_df, thresholds={'high':3.0, 'low':1.5}):
    """Generate enhanced signals with a moderate category."""
    date_col = next(col for col in predictions_df.columns if col.lower() == 'date')
    pred_series = predictions_df.set_index(date_col)['prediction']
    signals = pd.Series('neutral', index=pred_series.index)
    
    signals[pred_series > thresholds['high']] = 'high_inflation_warning'
    signals[(pred_series <= thresholds['high']) & (pred_series > thresholds['low'])] = 'moderate_inflation_warning'
    signals[pred_series < thresholds['low']] = 'low_inflation_warning'

    # Momentum signals (override previous if large momentum shifts occur)
    pred_momentum = pred_series.rolling(3).mean().diff()
    signals[pred_momentum > 0.5] = 'rapid_acceleration'
    signals[pred_momentum < -0.5] = 'rapid_deceleration'
    
    return signals

class ModelMonitor:
    """Monitor model performance and generate alerts."""
    def __init__(self, performance_window=12):
        self.performance_window = performance_window
        self.baseline_performance = None
        
    def update_metrics(self, new_predictions, actuals):
        error = np.abs(new_predictions - actuals)
        if self.baseline_performance is None:
            self.baseline_performance = error.mean()
            
        recent_performance = error.rolling(self.performance_window).mean()
        alerts = []
        
        if len(recent_performance.dropna()) > 0:
            if recent_performance.iloc[-1] > 1.5 * self.baseline_performance:
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'high',
                    'message': 'Model performance has degraded significantly'
                })

            pred_volatility = new_predictions.rolling(self.performance_window).std()
            if len(pred_volatility.dropna()) > 0:
                if pred_volatility.iloc[-1] > pred_volatility.quantile(0.95):
                    alerts.append({
                        'type': 'prediction_instability',
                        'severity': 'medium',
                        'message': 'Unusual volatility in predictions'
                    })
        
        return pd.DataFrame(alerts)

def main():
    logging.info("Starting inflation prediction system")
    
    try:
        data_loader = DataLoader('macro_data.csv')
        df = data_loader.load_data()
        if df.empty:
            raise ValueError("No data loaded")
        logging.info(f"Loaded data with shape: {df.shape}")
        
        fe = MacroFeatureEngineering(
            commodity_pca_components=2,
            momentum_windows=[3, 6, 12]
        )
        df = fe.fit_transform(df)
        logging.info(f"Completed feature engineering. Feature count: {len(df.columns)}")

        # Try regime detection
        regime_detector = RegimeDetector(k_regimes=2).fit(df)
        regime_probs = regime_detector.predict_regime_probs(df)

        logging.info("Starting backtesting")
        results_df = run_backtest(
            df=df,
            target_col='Core_PCE',
            horizons=[6, 9, 12],
            train_window=120
        )
        
        if len(results_df) == 0:
            raise ValueError("No results generated from backtesting")
        
        # Evaluate performance
        print("\nBacktest Results:")
        for h in [6, 9, 12]:
            subset = results_df[results_df['horizon'] == h].copy()
            if len(subset) > 0:
                rmse = np.sqrt(((subset['prediction'] - subset['actual'])**2).mean())
                mae = (subset['prediction'] - subset['actual']).abs().mean()
                coverage = ((subset['actual'] >= subset['lower_ci']) & 
                           (subset['actual'] <= subset['upper_ci'])).mean() * 100
                print(f"Horizon {h}M:")
                print(f"  RMSE: {rmse:.3f}")
                print(f"  MAE: {mae:.3f}")
                print(f"  Prediction Interval Coverage: {coverage:.1f}%")

        # Generate signals
        signals = create_inflation_signals(
            predictions_df=results_df,
            thresholds={'high': 3.0, 'low': 1.5}
        )
        print("\nSignal Distribution:")
        print(signals.value_counts())
        
        # Save results
        results_df.to_csv('backtest_results.csv', index=False)
        signals.to_csv('inflation_signals.csv')
        logging.info("Results and signals saved.")
        
        # Monitor
        monitor = ModelMonitor(performance_window=12)
        alerts = monitor.update_metrics(
            new_predictions=results_df['prediction'],
            actuals=results_df['actual']
        )

        if not alerts.empty:
            print("\nAlerts:")
            print(alerts)

        print("\nSummary Statistics:")
        print(f"Total Predictions: {len(results_df)}")
        print(f"Time Period: {results_df['Date'].min()} to {results_df['Date'].max()}")
        print(f"Average Error Std: {results_df['error'].std():.3f}")

        logging.info("Inflation prediction system completed successfully")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
    finally:
        logging.info("Inflation prediction system shutdown")

if __name__ == "__main__":
    main()

    """
    QA / TODO:
    
    The target_col parameter flow could be more consistent. 
    Currently: It's passed to some classes but not others
    Some classes have it hardcoded as 'Core_PCE'
    It should probably be a parameter at the top level of main()
    """
