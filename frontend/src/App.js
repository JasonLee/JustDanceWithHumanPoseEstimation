import React, { useState } from 'react';
import './App.css';
import SongList from './components/SongList';
import Login from './components/Login';

function App() {
  const [token, setToken] = useState();

  if (!token) {
    return <Login setToken={setToken} />
  }

  return (
    <div className="App">
      <header className="App-header">
        <SongList />
      </header>
    </div>
  );
}

export default App;