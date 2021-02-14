import React, { Component } from 'react';
import './App.css';
import SongList from './components/SongList';

class App extends Component {
    constructor(props) {
        super(props);
    }

    // What is actually displayed
    render() {
        return (
        <div className="App">
          <header className="App-header">
            <SongList/>
          </header>
        </div>
        );
    }
}

export default App;