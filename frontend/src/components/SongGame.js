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

        axios.post("/pose_score", {
            image: imageSrc,
        })
        .then(function (response) {
            //handle success
            console.log(response);
        })
        .catch(function (response) {
            //handle error
            console.log(response);
        });
    };

    playVideo = () => {
        this.videoRef.current.playing = !this.videoRef.playing;
    }

    componentDidMount() {
        // axios.get("/songs")
        //     .then(res => {
        //         let data = res.data;
        //         this.setState({ songs: data });
        //     }).catch((error) => {
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
                    <ReactPlayer controls={true} ref={this.videoRef} url='http://localhost:8000/songs/1' />
                    <button onClick={this.playVideo}>Play/Pause</button>
                </div>
            </>
        );
    }
}
