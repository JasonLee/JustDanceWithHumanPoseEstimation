import React, { Component } from 'react';
import { withRouter } from "react-router";
import Webcam from "react-webcam";
import ReactPlayer from 'react-player'
import axios from 'axios';
import ScoreCanvas from './ScoreCanvas';

const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: "user"
};

class SongGame extends Component {
    constructor(props) {
        super(props);
        this.history = this.props.history;

        this.state = {
            joints: [],
            mapping: [],
            truth_joints: [],
            current_time: 0,
            gameID: "",
            videoLoaded: false,
            webcamLoaded: false,
            playing:true,
            score: 0
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
            songName: "test",
            gameID: this.state.gameID,
            songID: this.state.songID

        })
        .then(response => {
            this.setState({
                            "joints": response.data.joints,
                            "mapping": response.data.mapping,
                            "truth_joints": response.data.truth_joints,
                            "results": response.data.results,
                            "score": this.state.score + response.data.score
                          })

        })
        .catch(error => {
            console.log(error);
        });
        
    };

    playing = ({ playedSeconds, loadedSeconds}) => {
        // console.log("playedSeconds", playedSeconds, Math.round(playedSeconds));
        this.setState({"current_time": playedSeconds})

        // if (Math.round(playedSeconds) % 1 == 0) {
        this.capture();
        // }
    }

    pause = () => {
        this.capture()
    }

    // Song/Game has ended. Redirects user to results screen
    ended = () => {
        this.history.push("/results/" + this.state.gameID);
    }

    onBufferEnd = () => {
        this.setState({videoLoaded: true})
    }

    componentDidMount() {
        axios.post("/create_game")
            .then(res => {
                let data = res.data.gameID;
                this.setState({ gameID: data });
                console.log("gameID", data);
            }).catch(error => {
                console.log(error);
            });
    };

    // What is actually displayed
    render() {

        return ( 
            <>
                { (!this.state.webcamLoaded && !this.state.videoLoaded)? "Loading" : "Done" } <br/>
                Score: {this.state.score}
                <div>
                    <Webcam onUserMedia={() => {this.setState({webcamLoaded: true})}} videoConstraints={videoConstraints} ref={this.webcamRef}/>
                    <button onClick={this.capture}>Capture photo</button>
                    <ScoreCanvas key={this.state.joints} joints={this.state.joints} mapping={this.state.mapping} results={this.state.results}/>
                    <ScoreCanvas key={this.state.truth_joints} joints={this.state.truth_joints} mapping={this.state.mapping} results={this.state.results}/>
                    
                </div>
                <div>
                    <ReactPlayer onBufferEnd={this.onBufferEnd} 
                        onReady={() => {this.setState({videoLoaded: true})}}
                        controls={true} 
                        onEnded={this.ended} 
                        playing={this.state.playing} 
                        muted={true} 
                        progressInterval={500} 
                        onProgress={this.playing} 
                        onPause={this.pause}
                        ref={this.videoRef} 
                        url='http://localhost:8000/songs/1' />
                </div>
            </>
        );
    }
}

export default withRouter(SongGame);