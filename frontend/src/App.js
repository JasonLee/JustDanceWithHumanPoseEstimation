import React, { Component } from 'react';
import logo from './logo.svg';
import axios from 'axios';
import './App.css';

class App extends Component {
    constructor(props) {
        super(props);
        this.state  = {
            test: [],
        };
    }

    // Runs when component has been added
    componentDidMount() {
        axios.get("/test").then(res => {
            let data = res.data;
            console.log(data)
            this.setState({test: data});
        })
    };

    // What is actually displayed
    render() {
        return (
        <div className="App">
          <header className="App-header">
            <p>
              {this.state.test}
            </p>
          </header>
        </div>
        );
    }
}

export default App;