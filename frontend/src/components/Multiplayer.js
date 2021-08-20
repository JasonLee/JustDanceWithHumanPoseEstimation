import React, { Component } from 'react';
import { withRouter } from "react-router"
import { Link } from 'react-router-dom';
import { io } from "socket.io-client";
import { List, ListItem, Button } from '@material-ui/core/';
import styles from './css/Multiplayer.module.css'

import ChatBox from './ChatBox';
import MultiplayerSelectedSong from './MultiplayerSelectedSong';
import MultiplayerPlayerBox from './MultiplayerPlayerBox';

class Multiplayer extends Component {
    MAXPLAYERS = 10;

    constructor(props) {
        super(props);

        this.state = {
            text: "",
            messages: [""],
            players: Array(8).fill(null),
            lobbyID: props.match.params.lobbyID,
            songID: props.match.params.songID,
            songData: [],
            token: props.token,
            MAX_PLAYERS: 5
        };

        this.constructPlayerList();
        this.socket = io('/');
        this.history = props.history;
    }

    componentDidMount() {
        this.socket.emit('host', { 
            token: this.state.token, 
            lobbyID: this.state.lobbyID,
        });

        // When an event message is received, update UI or start game
        this.socket.on("event", message => {
            if (message == "START") {
                this.history.push(`/game/${this.state.songID}/${this.state.lobbyID}`);
            }else if(typeof message === 'number') {
                console.log("CHANGED TO", message);
                this.setState({songID: message})
            }
        });

        if(this.state.songID) {
            this.socket.emit('event', { 
                token: this.state.token, 
                lobbyID: this.state.lobbyID,
                songID: this.state.songID,
                start: false
            });
        }

    };
    
    // Update player list 
    constructPlayerList(players) {
        let rows = []

        for(let i=0; i< this.MAXPLAYERS; i++) {
            if(!players || !players[i]) {
                rows.push(<ListItem key={i}> Test</ListItem>)
            }else{
                rows.push(<ListItem key={i}> {players[i]}</ListItem>)
            }
        }
        
        return rows
    }

    // Start game event to all clients
    start = (event) => {
        this.socket.emit('event', { 
            token: this.state.token, 
            lobbyID: this.state.lobbyID,
            songID: this.state.songID,
            start: true
        });

    }

    // Chatbox callback for new users
    outputUsers = (users) => {
        if (!users) {
            return;
        }
        
        while(users.length < this.state.MAX_PLAYERS) {
            users.push(null);
        }
        
        console.log("PLAYER LIST", users);
        this.setState({players: users})
        this.forceUpdate();
    }

    render() {
        return (
            <div className={styles.Container}>
                <div className={styles.multiContainer}>
                    <List>
                        {this.state.players.map((player, i) => 
                            <ListItem key={player ? player.id : i}> {player ? <MultiplayerPlayerBox username={player.username} token={this.state.token}/> 
                            : 
                            <div className={styles.none}>NONE</div>} </ListItem>
                        )}
                    </List>

                    <ChatBox lobbyID={this.state.lobbyID} token={this.state.token} outputUsers={this.outputUsers} socket={this.socket} />
                </div>
                

                <div>   
                    <div>
                        {this.state.songID ? <MultiplayerSelectedSong token={this.state.token} songID={this.state.songID} /> : <div style={{color: "white"}}>No Song Selected</div> } <br />
                        <Link to={"/songs/"+ this.state.lobbyID}>
                            <Button variant="contained" color="primary"> Select Song </Button>
                        </Link>
                    </div>
                    <br />
                    {this.state.songID && <div>
                        <Link to={`/game/${this.state.songID}/${this.state.lobbyID}`}>
                            <Button onClick={this.start} variant="contained" color="primary"> Start Game </Button>
                        </Link>
                    </div>}
                </div>
            </div>
        );
    }
}

export default withRouter(Multiplayer);