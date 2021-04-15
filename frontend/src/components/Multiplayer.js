import React, { Component } from 'react';
import axios from 'axios';
import { withRouter } from "react-router"
import { Link } from 'react-router-dom';
import { List, ListItem, ListItemText, Paper, Grow } from '@material-ui/core/';

import ChatBox from './ChatBox';

class Multiplayer extends Component {
    constructor(props) {
        super(props);

        this.state = {
            text:"",
            messages: [""],
            players: ["a","a"],
            lobbyID: props.match.params.lobbyID,
            songID: props.match.params.songID
        };
        this.constructPlayerList();

        const token = props.token;
        console.log("token", token);
    }

    componentDidMount() {
    };

    constructPlayerList(players) {
        const MAXPLAYERS = 10;
        let rows = []

        for(let i=0; i<MAXPLAYERS; i++) {
            if(!players || !players[i]) {
                rows.push(<ListItem key={i}> Test</ListItem>)
            }else{
                rows.push(<ListItem key={i}> {players[i]}</ListItem>)
            }
        }
        
        return rows
    }

    outputUsers = (users) => {
        this.setState({players: users})

        // console.log("players", this.state.players)
        this.players = users;

        console.log("players", this.players)
        this.forceUpdate();
    }

    startGame = (event) => {
        event.preventDefault();
    }

    render() {
        return (
            <div>
                <List>
                    {this.state.players.map((player, i) => 
                        <ListItem key={i}> {player.username} </ListItem>
                    )}
                </List>
                <div>
                    <Link to={"/songs/"+ this.state.lobbyID}>
                        <button> Select Song </button>
                    </Link>
                </div>

                <div>
                    <Link to={`/game/${this.state.songID}/${this.state.lobbyID}`}>
                        <button> Start Game </button>
                    </Link>
                </div>
                <ChatBox lobbyID={this.state.lobbyID} outputUsers={this.outputUsers} />
            </div>
        );
    }
}

export default withRouter(Multiplayer);