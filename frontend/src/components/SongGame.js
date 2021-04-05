import React, { Component } from 'react';
import Webcam from "react-webcam";
import ReactPlayer from 'react-player'
import axios from 'axios';
import ScoreCanvas from './ScoreCanvas';

const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
};

export default class SongGame extends Component {
    constructor(props) {
        super(props);
        this.state = {
            "joints": [],
            "mapping": [],
            "truth_joints": [],
            "current_time": 0
        };
        
        this.webcamRef = React.createRef();
        this.videoRef = React.createRef();
        
    }

    capture = () => {
        const imageSrc = this.webcamRef.current.getScreenshot();

        if (imageSrc) {
            console.log("SCREENSHOT");
        }
        
        axios.post("/pose_score", {
            image: imageSrc,
            timestamp: this.state.current_time * 100,
            songName: "test"

        })
        .then(response => {
            //handle success
            console.log("response", response.data)
            this.setState({
                            "joints": response.data.joints,
                            "mapping": response.data.mapping,
                            "truth_joints": response.data.truth_joints
                          })

        })
        .catch(error => {
            console.log(error);
        });
        
    };

    playVideo = () => {
        this.videoRef.current.pause();
    }

    playing = ({ playedSeconds, loadedSeconds}) => {
        // console.log("playedSeconds", playedSeconds, Math.round(playedSeconds));
        this.setState({"current_time": playedSeconds})
        // console.log("loadedSeconds", loadedSeconds, Math.round(loadedSeconds));

        // if (Math.round(playedSeconds) % 1 == 0) {
        //     this.capture();
        // }
    }

    pause = () => {
        this.capture()
    }

    componentDidMount() {
        // axios.get("/songs")
        //     .then(res => {
        //         let data = res.data;
        //         this.setState({ songs: data });
        //     }).catch(error => {
        //         console.log(error);
        //     });
    };

    // What is actually displayed
    render() {
        return ( 
            <>
                <div>
                    <Webcam videoConstraints={videoConstraints} ref={this.webcamRef}/>
                    <button onClick={this.capture}>Capture photo</button>
                    <ScoreCanvas key={this.state.joints} joints={this.state.joints} mapping={this.state.mapping}/>
                    <ScoreCanvas key={this.state.truth_joints} joints={this.state.truth_joints} mapping={this.state.mapping}/>
                    
                </div>
                <div>
                    <ReactPlayer controls={true} playing={true} muted={true} progressInterval="500" onProgress={this.playing} onPause={this.pause}ref={this.videoRef} url='http://localhost:8000/songs/1' />
                    <button onClick={this.playVideo}>Play/Pause</button>
                </div>
            </>
        );
    }
}
