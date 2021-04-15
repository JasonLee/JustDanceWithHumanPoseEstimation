import React, { Component } from 'react';
import axios from 'axios';
import { List, ListItem, ListItemText, Paper, Grow } from '@material-ui/core/';


export default class MultiplayerScoreList extends Component {
    interval = null;

    constructor(props) {
        super(props);
        
        this.state = {
            lobbyID: props.lobbyID,
            scores: []
        }
    }


    componentDidMount() {
        this.interval = setInterval(this.getData, 10000);
        this.getData();
    }

    componentWillUnmount() {
        clearInterval(this.interval);
    }

    getData = () => {

        console.log("SCORES LOBBY ID", this.state.lobbyID)
        axios.get("/getLobbyScores", {
            params: {
                lobbyID: this.state.lobbyID
            }
        }).then(res => {
            console.log("MULT", res.data)
            this.setState({ scores: res.data})
        });
    };

    render() {
        return (
            <div>
                <List>
                    {this.state.scores.sort().map( (data, i) => 
                        <ListItem key={i}> 
                            {data.playerID} <br />
                            {data.score}
                        </ListItem>
                    )}
                </List>
            </div>
        );
    }
}