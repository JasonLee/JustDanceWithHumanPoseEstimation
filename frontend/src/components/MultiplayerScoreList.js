import React, { Component } from 'react';
import axios from 'axios';
import { List, ListItem } from '@material-ui/core/';
import styles from './css/MultiplayerScoreList.module.css';

// Multiplayer in game score list
export default class MultiplayerScoreList extends Component {
    interval = null;

    constructor(props) {
        super(props);
        
        this.state = {
            gameID: props.gameID,
            lobbyID: props.lobbyID,
            data: [],
            token: props.token
        }
        
        this.currentScoreRef = React.createRef();
    }
        
    componentDidUpdate() {
        // Scroll user score in view
        if (this.currentScoreRef.current) {
            this.currentScoreRef.current.scrollIntoView(
                {
                    behavior: 'auto',
                    block: 'center',
                    inline: 'center'
                }
            );
        }
    }


    componentDidMount() {
        // Update every 10 seconds
        this.interval = setInterval(this.getData, 10000);
        this.getData();
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    getData = () => {
        axios.get("/getLobbyScores", {
            params: {
                lobbyID: this.state.lobbyID
            },
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(res => {
            this.setState({ data: res.data})
        });
    };

    render() {
        return (
            <div>
                <div className={styles.Paper} styles={{ overflow: "auto" }}>
                    <List styles={{ overflow: "auto" }}>
                        {this.state.data.sort().map( (playerDetails, i) => 
                            <ListItem key={i} ref={playerDetails.gameID == this.state.gameID ? this.currentScoreRef : null}> 
                                <div className={styles.playerWrapper}>
                                    <b style={{paddingRight: "5px"}}> {i}. </b>
                                    <img className={styles.image} src={playerDetails.user.profilePic} />
                                    <div className={styles.user}>{playerDetails.user.username} </div> 
                                    <div> {playerDetails.score} </div>
                                </div>
                            </ListItem>
                        )}
                    </List>
                </div>
            </div>
        );
    }
}