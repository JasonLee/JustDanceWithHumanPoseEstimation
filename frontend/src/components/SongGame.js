import React, { Component } from 'react';
import { withRouter } from "react-router";
import Webcam from "react-webcam";
import ReactPlayer from 'react-player'
import axios from 'axios';
import ScoreCanvas from './ScoreCanvas';
import MultiplayerScoreList from './MultiplayerScoreList';
import ScoreList from './ScoreList';
import styles from './css/SongGame.module.css';

const videoConstraints = {
    width: { min: 480 },
    height: { min: 720 },
    aspectRatio: 0.6666666667
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
            webcamLoaded: false,
            playing:true,
            score: 0,
            gameID: "1",
            songID: parseInt(props.match.params.songID),
            lobbyID: props.match.params.lobbyID,
            token: props.token,
            progress: 0,
            prevScores: []
        };

        this.webcamRef = React.createRef();
        this.videoRef = React.createRef();
    }

    // Function to submit the image to evaluate
    capture = () => {
        const imageSrc = this.webcamRef.current.getScreenshot();
        
        axios.post("/pose_score", {
            image: imageSrc,
            timestamp: this.state.current_time * 100,
            songName: "test",
            gameID: this.state.gameID,
            songID: this.state.songID
        },{            
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(response => {
            this.setState({
                "joints": response.data.joints,
                "mapping": response.data.mapping,
                "truth_joints": response.data.truth_joints,
                "results": response.data.results,
                "score": this.state.score + response.data.score
            });

        })
        .catch(error => {
            console.error(error);
        });
        
    };

    // Called by video player every "progressInterval" milliseconds (500)
    playing = ({ played, playedSeconds, loadedSeconds}) => {
        this.setState({"current_time": playedSeconds, progress: Math.round(played * 100)})
        this.capture();
    }

    pause = () => {
        this.capture()
    }

    // Song/Game has ended. Redirects user to results screen
    ended = () => {
        axios.post("/finishGame", {
            gameID: this.state.gameID
        },{            
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(response => {
            // Redirect to results for multiplayer lobby
            if(this.state.lobbyID == undefined || this.state.lobbyID == null) {
                this.history.replace({
                    pathname: `/results/${this.state.gameID}`, 
                    state: { from: this.props.location }
                });
            }else{
                // Redirect to results for multiplayer lobby
                this.history.replace({
                    pathname: `/results/${this.state.gameID}/${this.state.lobbyID}`, 
                    state: { from: this.props.location }
                });
            }
            
        })
        .catch(error => {
            console.log(error);
        });
    }

    componentDidMount() {
        axios.post("/createGame", {
            songID: this.state.songID,
            lobbyID: this.state.lobbyID
        }, {            
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(res => {
                let data = res.data.gameID;
                this.setState({ gameID: data });
        }).catch(error => {
            console.error(error);
        });

        axios.get('/getUserSongScores', {
            headers: {
                "Authorization" : `Bearer ${this.state.token}`
            },
            params: {
                "songID": this.state.songID
            }
        }).then(response => {
            this.setState({ prevScores:  response.data});
        });
    };

    // What is actually displayed
    render() {
        return ( 
            <div>
                <div className={styles.ScoreWrapper}> 
                    {this.state.lobbyID ? 
                        <MultiplayerScoreList token={this.state.token} gameID={this.state.gameID} lobbyID={this.state.lobbyID} /> : <ScoreList key={this.state.score} score={this.state.score} prevScores={this.state.prevScores} />
                    }
                    {/* <button onClick={this.capture}>Capture photo</button> */}
                </div>

                <div className={styles.progressWrapper}>
                    <div key={this.state.progress} className={styles.progress} style={{"width" : ""+this.state.progress+"%"}}>{this.state.progress + "%"}</div>
                </div>
                
                <div className={styles.Container}>
                    <div className={styles.WebcamWrapper} >
                        <Webcam 
                            className={styles.Webcam} 
                            onUserMedia={() => {this.setState({webcamLoaded: true})}} 
                            videoConstraints={videoConstraints} 
                            ref={this.webcamRef}
                        />
                    </div>

                    <div className={styles.VideoWrapper}>
                        <ReactPlayer onBufferEnd={this.onBufferEnd} 
                            controls={true} 
                            onEnded={this.ended} 
                            playing={false}//{this.state.playing}
                            muted={false} 
                            progressInterval={500} 
                            onProgress={this.playing} 
                            onPause={this.pause}
                            ref={this.videoRef} 
                            url='http://localhost:8000/songs/1'
                            width={480} 
                            height={720}
                            className={styles.VideoPlayer}
                            />
                    </div>
                </div>
                <ScoreCanvas className={styles.PlayerJoints} key={1} truth={false} joints={this.state.joints} mapping={this.state.mapping} results={this.state.results}/>
                <ScoreCanvas className={styles.TruthJoints} key={2} truth={true} joints={this.state.truth_joints} mapping={this.state.mapping} results={this.state.results}/>
            </div>
        );
    }
}

export default withRouter(SongGame);