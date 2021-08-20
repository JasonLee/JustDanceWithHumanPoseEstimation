import React, { Component } from 'react';
import UIfx from 'uifx';
import slider from './soundFX/slider.mp3'

// Component to handle all audio in Song list cards
export default class Music extends Component {
    constructor(props) {
        super(props);
        this.state = {
            play: false,
            id: props.id,
            volume: 0.1
        }
        
        this.audio = new Audio("http://127.0.0.1:8000/audio");
        this.audio.volume = 0.1;
        this.audio.loop = true;

        this.changeVolume = this.changeVolume.bind(this);

        this.tick = new UIfx(slider,
            {
              volume: 0.5, // value must be between 0.0 ⇔ 1.0
              throttleMs: 50
            }
        )
    }

    componentDidMount() {
        this.play();
    }

    componentWillUnmount() {
        this.audio.pause();
        this.audio.currentTime = 0
    }

    play = () => {
        this.setState({ play: true, pause: false })
        this.audio.play();
    }

    changeVolume = (event) => {
        this.tick.play();
        this.setState({volume: event.target.value});
        this.audio.volume = event.target.value;
    }

    render() {
        return (
            <div>
                <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.02}
                    value={this.state.volume}
                    onChange={this.changeVolume}
                    />
            </div>
        );
    }
}
