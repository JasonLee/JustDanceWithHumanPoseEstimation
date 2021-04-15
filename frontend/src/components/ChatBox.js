import React, { Component } from 'react';
import axios from 'axios';
import { List, ListItem, ListItemText, Paper, Grow } from '@material-ui/core/';
import { io } from "socket.io-client";

class ChatBox extends Component {
    constructor(props) {
        super(props);
        this.state = {
            text:"",
            messages:[""],
            lobbyID: props.lobbyID,
            username: "Test"
        };

        this.sendMessage = this.sendMessage.bind(this);
        this.inputBox = React.createRef();
        this.outputUsers = props.outputUsers;
    }

    componentDidMount() {
        this.socket = io('/',{ 
            query: "bearer=token" + "&" + "lobbyID=" + this.state.lobbyID
        });

        this.socket.emit('joinRoom', { 
            username: this.state.username, 
            lobbyID: this.state.lobbyID 
        });

        this.socket.on("message", message => {
            const joined = this.state.messages.concat(message);
            this.setState({ messages: joined })
        });

        // Get room and users
        this.socket.on('roomUsers', ({ users }) => {
            this.outputUsers(users);
        });

    };

    componentWillUnmount() {
        this.socket.disconnect();
    }

    sendMessage = (event) => {
        event.preventDefault();
        let message = this.inputBox.current.value

        this.socket.emit("chatMessage",  message);

        this.inputBox.current.value = ""

    }

    render() {
        return (
            <div>
                <form>
                    <List>
                        {this.state.messages.map((message, i) => 
                            <ListItemText key={i} primary={message}></ListItemText>
                        )}

                    </List>
                    <input id="message" className="message" placeholder="Message" ref={this.inputBox}/>
                    <button onClick={this.sendMessage}>Send</button>
                </form>
            </div>
        );
    }
}

export default ChatBox;