import React, { Component } from 'react';
import Webcam from "react-webcam";
import ReactPlayer from 'react-player'
import axios from 'axios';

const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
};

export default class SongGame extends Component {
   
    constructor(props) {
        super(props);

        this.webcamRef = React.createRef();
        this.videoRef = React.createRef();
        
    }

    capture = () => {
        const imageSrc = this.webcamRef.current.getScreenshot();

        if (imageSrc) {
            console.log("SCREENSHOT");
        }else{
            console.log("fucked");
        }   
        
        axios.post("/pose_score", {
            image: imageSrc,
        })
        .then(response => {
            //handle success
            console.log(response);
        })
        .catch(error => {
            console.log(error);
        });
        
    };

    playVideo = () => {
        // this.videoRef.current.play();
    }

    playing = ({ playedSeconds }) => {
        console.log("playedSeconds",Math.round(playedSeconds));
        if (Math.round(playedSeconds) % 1 == 0) {
            this.capture();
        }
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
                    
                </div>
                <div>
                    <ReactPlayer playing={true} onProgress={this.playing} ref={this.videoRef} url='http://localhost:8000/songs/1' />
                    <button onClick={this.playVideo}>Play/Pause</button>
                </div>
            </>
        );
    }
}
