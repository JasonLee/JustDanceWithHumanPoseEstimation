import React, { useState } from 'react';
import './App.css';
import SongList from './components/SongList';
import Login from './components/Login';
import Register from './components/Register';

function App() {
  const [token, setToken] = useState();

  if (!token) {
    return <Register setToken={setToken} />
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