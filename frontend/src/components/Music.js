import React, { Component } from 'react';

export default class Music extends Component {
    constructor(props) {
        super(props);
        this.state = {
            play: false
        }
        
        this.audio = new Audio("http://127.0.0.1:8000/audio2");
        this.audio.volume = 0.1;
        // console.log("Music");
    }

    componentDidMount() {
        this.play();
    }

    componentWillUnmount() {
        this.audio.pause();
        this.audio.currentTime = 0
    }

    play = () => {
        console.log("Play")
        this.setState({ play: true, pause: false })
        this.audio.play();
    }

    pause = () => {
        console.log("Pause")
        this.setState({ play: false, pause: true })
        this.audio.pause();
    }

    render() {
        return (
            <div>
                <button onClick={this.pause}>Pause</button>
            </div>
        );
    }
}
