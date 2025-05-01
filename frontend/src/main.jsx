import React from 'react'; // Import React
import ReactDOM from 'react-dom/client'; // Correct import for React 18+
import './index.css' // Keep index.css import
import App from './App'; // Import App component

// Use createRoot for React 18+
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode> {/* Wrap with StrictMode */}
    <App />
  </React.StrictMode>,
);
