import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Prediction state
  const [features, setFeatures] = useState(Array(10).fill(0));
  const [prediction, setPrediction] = useState(null);
  
  // Model state
  const [modelInfo, setModelInfo] = useState(null);
  const [allMetrics, setAllMetrics] = useState(null);
  
  // Drift state
  const [driftReport, setDriftReport] = useState(null);
  const [driftHistory, setDriftHistory] = useState([]);
  
  // Stats state
  const [predictionStats, setPredictionStats] = useState(null);
  const [retrainHistory, setRetrainHistory] = useState(null);

  // Check health and load model
  useEffect(() => {
    checkHealth();
    fetchModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setModelLoaded(data.model_loaded);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model/info`);
      if (response.ok) {
        const data = await response.json();
        setModelInfo(data);
      }
    } catch (error) {
      console.error('Failed to fetch model info:', error);
    }
  };

  // ========== TRAINING FUNCTIONS ==========

  const trainModel = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/model/train?n_samples=1000`, {
        method: 'POST'
      });
      const data = await response.json();
      
      if (response.ok) {
        alert('✓ Model training completed successfully!');
        setModelLoaded(true);
        await fetchModelInfo();
        await fetchModelComparison();
      } else {
        alert(`✗ Training failed: ${data.detail}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const retrain = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/model/retrain?force=true`, {
        method: 'POST'
      });
      const data = await response.json();
      alert(data.message);
      await fetchModelInfo();
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ========== PREDICTION FUNCTIONS ==========

  const handleFeatureChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = parseFloat(value) || 0;
    setFeatures(newFeatures);
  };

  const makePrediction = async () => {
    if (!modelLoaded) {
      alert('Please train model first');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: features,
          feature_names: [
            'skill_overlap_score', 'experience_years', 'certification_match',
            'location_proximity', 'salary_fit', 'education_level',
            'soft_skills_match', 'project_relevance', 'growth_potential', 'cultural_fit'
          ]
        })
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      alert(`Prediction failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ========== DRIFT FUNCTIONS ==========

  const checkDrift = async () => {
    if (!modelLoaded) return;
    
    setLoading(true);
    try {
      // Generate random drift test data
      const testData = Array(5).fill(0).map(() => 
        Array(10).fill(0).map(() => Math.random() * 2 - 1)
      );

      const response = await fetch(`${API_BASE_URL}/drift/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: testData })
      });
      const data = await response.json();
      setDriftReport(data);
      setDriftHistory(prev => [data, ...prev.slice(0, 4)]);
    } catch (error) {
      alert(`Drift check failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ========== METRICS FUNCTIONS ==========

  const fetchModelComparison = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats/model-comparison`);
      const data = await response.json();
      setAllMetrics(data.model_comparison);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const fetchStats = async () => {
    try {
      const [predResponse, retrainResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/stats/predictions?limit=50`),
        fetch(`${API_BASE_URL}/stats/retrain-history`)
      ]);

      const predData = await predResponse.json();
      const retrainData = await retrainResponse.json();

      setPredictionStats(predData);
      setRetrainHistory(retrainData);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  // ========== RENDER FUNCTIONS ==========

  const renderTrainingTab = () => (
    <div className="tab-content">
      <h2>Model Training & Management</h2>
      <div className="button-group">
        <button 
          onClick={trainModel} 
          disabled={loading}
          className="btn btn-primary"
        >
          {loading ? 'Training...' : 'Train Model'}
        </button>
        <button 
          onClick={retrain}
          disabled={loading || !modelLoaded}
          className="btn btn-secondary"
        >
          {loading ? 'Retraining...' : 'Force Retrain'}
        </button>
      </div>

      {modelInfo && (
        <div className="info-box">
          <h3>✓ Model Information</h3>
          <p><strong>Best Model:</strong> {modelInfo.best_model}</p>
          <p><strong>Models Available:</strong> {modelInfo.models_available.join(', ')}</p>
          <p><strong>Training Time:</strong> {new Date(modelInfo.training_timestamp).toLocaleString()}</p>
          
          <h4>Model Metrics:</h4>
          <div className="metrics-grid">
            {modelInfo.metrics && modelInfo.metrics[modelInfo.best_model] && (
              <>
                <div className="metric-card">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].accuracy * 100).toFixed(2)}%</div>
                  <div className="metric-label">Accuracy</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].precision * 100).toFixed(2)}%</div>
                  <div className="metric-label">Precision</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].recall * 100).toFixed(2)}%</div>
                  <div className="metric-label">Recall</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].f1_score * 100).toFixed(2)}%</div>
                  <div className="metric-label">F1 Score</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].auc_roc * 100).toFixed(2)}%</div>
                  <div className="metric-label">AUC-ROC</div>
                </div>
                <div className="metric-card danger">
                  <div className="metric-value">{(modelInfo.metrics[modelInfo.best_model].false_positive_rate * 100).toFixed(2)}%</div>
                  <div className="metric-label">False Positive Rate</div>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {allMetrics && (
        <div className="comparison-box">
          <h3>Model Comparison</h3>
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>AUC-ROC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(allMetrics).map(([name, metrics]) => (
                <tr key={name} className={name === modelInfo?.best_model ? 'best-model' : ''}>
                  <td>{name}</td>
                  <td>{(metrics.accuracy * 100).toFixed(2)}%</td>
                  <td>{(metrics.precision * 100).toFixed(2)}%</td>
                  <td>{(metrics.recall * 100).toFixed(2)}%</td>
                  <td>{(metrics.f1_score * 100).toFixed(2)}%</td>
                  <td>{(metrics.auc_roc * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );

  const renderPredictionTab = () => (
    <div className="tab-content">
      <h2>Make Predictions</h2>
      {!modelLoaded && <div className="warning">⚠️ Train model first</div>}
      
      <div className="features-input">
        <h3>Input Features</h3>
        <div className="features-grid">
          {[
            'Skill Overlap', 'Experience', 'Certification', 'Location',
            'Salary Fit', 'Education', 'Soft Skills', 'Project Match',
            'Growth Potential', 'Cultural Fit'
          ].map((label, idx) => (
            <div key={idx} className="feature-input">
              <label>{label}</label>
              <input
                type="number"
                value={features[idx]}
                onChange={(e) => handleFeatureChange(idx, e.target.value)}
                step="0.1"
                min="-3"
                max="3"
              />
            </div>
          ))}
        </div>
        <button 
          onClick={makePrediction}
          disabled={loading || !modelLoaded}
          className="btn btn-primary"
        >
          {loading ? 'Predicting...' : 'Predict Match'}
        </button>
      </div>

      {prediction && (
        <div className={`prediction-result ${prediction.match ? 'match' : 'no-match'}`}>
          <h3>Prediction Result</h3>
          <div className="result-content">
            <div className="match-status">
              {prediction.match ? '✓ MATCH FOUND' : '✗ NO MATCH'}
            </div>
            <div className="confidence">
              Confidence: <strong>{(prediction.confidence * 100).toFixed(2)}%</strong>
            </div>
            <div className="model-info">
              Model: <strong>{prediction.model_used}</strong>
            </div>
            <div className="explanation">
              <strong>Explanation:</strong> {prediction.explanation}
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderDriftTab = () => (
    <div className="tab-content">
      <h2>Drift Monitoring & Detection</h2>
      <div className="button-group">
        <button 
          onClick={checkDrift}
          disabled={loading || !modelLoaded}
          className="btn btn-primary"
        >
          {loading ? 'Checking...' : 'Check for Drift'}
        </button>
      </div>

      {driftReport && (
        <div className={`drift-report ${driftReport.drift_detected ? 'alert' : 'safe'}`}>
          <h3>Drift Detection Report</h3>
          <div className="drift-content">
            <div className={`drift-status ${driftReport.drift_detected ? 'detected' : 'none'}`}>
              {driftReport.drift_detected ? '⚠️ DRIFT DETECTED' : '✓ NO DRIFT'}
            </div>
            <div className="drift-metrics">
              <p><strong>Drift Magnitude:</strong> {driftReport.drift_magnitude.toFixed(4)}</p>
              <p><strong>Affected Features:</strong> {driftReport.affected_features}</p>
              <p><strong>Threshold:</strong> {driftReport.threshold}</p>
              <p><strong>Time:</strong> {new Date(driftReport.timestamp).toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}

      {driftHistory.length > 0 && (
        <div className="history-box">
          <h3>Drift History</h3>
          <div className="history-list">
            {driftHistory.map((drift, idx) => (
              <div key={idx} className={`history-item ${drift.drift_detected ? 'alert' : ''}`}>
                <span>{new Date(drift.timestamp).toLocaleString()}</span>
                <span className={drift.drift_detected ? 'detected' : 'none'}>
                  {drift.drift_detected ? 'Drift' : 'Normal'} (Mag: {drift.drift_magnitude.toFixed(3)})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderMonitoringTab = () => (
    <div className="tab-content">
      <h2>System Monitoring</h2>
      <div className="button-group">
        <button 
          onClick={fetchStats}
          className="btn btn-primary"
        >
          Refresh Statistics
        </button>
      </div>

      {predictionStats && (
        <div className="stats-box">
          <h3>Prediction Statistics</h3>
          <p><strong>Total Predictions:</strong> {predictionStats.total_predictions}</p>
          <div className="recent-predictions">
            <h4>Recent Predictions ({predictionStats.recent_predictions.length})</h4>
            {predictionStats.recent_predictions.slice(-5).reverse().map((pred, idx) => (
              <div key={idx} className="prediction-log">
                <span>{new Date(pred.timestamp).toLocaleTimeString()}</span>
                <span>{pred.prediction.match ? '✓' : '✗'}</span>
                <span>{(pred.prediction.confidence * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {retrainHistory && (
        <div className="stats-box">
          <h3>Retraining History</h3>
          <p><strong>Total Retrains:</strong> {retrainHistory.total_retrains}</p>
          {retrainHistory.history.length > 0 && (
            <div className="retrain-list">
              {retrainHistory.history.map((retrain, idx) => (
                <div key={idx} className="retrain-item">
                  <p><strong>Time:</strong> {new Date(retrain.timestamp).toLocaleString()}</p>
                  <p><strong>Reason:</strong> {retrain.reason}</p>
                  <p><strong>Model:</strong> {retrain.best_model}</p>
                  <p><strong>F1 Score:</strong> {(retrain.best_f1_score * 100).toFixed(2)}%</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="App">
      <header className="app-header">
        <h1>🎯 PlaceMux ML Dashboard</h1>
        <p>Job-Skill Matching with Drift Detection & Auto-Retraining</p>
        <div className="status-indicator">
          {modelLoaded ? '✓ Model Ready' : '✗ Model Not Loaded'}
        </div>
      </header>

      <nav className="tab-navigation">
        <button 
          className={`tab-btn ${activeTab === 'train' ? 'active' : ''}`}
          onClick={() => { setActiveTab('train'); fetchModelComparison(); }}
        >
          📊 Training
        </button>
        <button 
          className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
          onClick={() => setActiveTab('predict')}
        >
          🔮 Predict
        </button>
        <button 
          className={`tab-btn ${activeTab === 'drift' ? 'active' : ''}`}
          onClick={() => setActiveTab('drift')}
        >
          📈 Drift
        </button>
        <button 
          className={`tab-btn ${activeTab === 'monitor' ? 'active' : ''}`}
          onClick={() => { setActiveTab('monitor'); fetchStats(); }}
        >
          📉 Monitor
        </button>
      </nav>

      <main className="app-main">
        {activeTab === 'train' && renderTrainingTab()}
        {activeTab === 'predict' && renderPredictionTab()}
        {activeTab === 'drift' && renderDriftTab()}
        {activeTab === 'monitor' && renderMonitoringTab()}
      </main>

      <footer className="app-footer">
        <p>PlaceMux © 2024 | ML Pipeline with Drift Detection & Auto-Retraining</p>
      </footer>
    </div>
  );
}

export default App;
