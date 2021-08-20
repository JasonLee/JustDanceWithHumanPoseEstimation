import React, { Component } from 'react';
import { List, ListItemText, Paper, Grow, Button } from '@material-ui/core/';
import styles from "./css/ChatBox.module.css";

// Chatbox for Multiplayer Feature
class ChatBox extends Component {
    constructor(props) {
        super(props);
        this.state = {
            text:"",
            messages:[""],
            lobbyID: props.lobbyID,
            username: "Test",
            token: props.token
        };

        this.socket = props.socket
        this.sendMessage = this.sendMessage.bind(this);
        this.inputBox = React.createRef();
        this.bottomMsg = React.createRef();
        this.outputUsers = props.outputUsers;
    };

    componentDidMount() {
        // User joins room
        this.socket.emit('joinRoom', { 
            token: this.state.token, 
            lobbyID: this.state.lobbyID,
        });

        // When a message is received, add to list display
        this.socket.on("message", message => {
            const joined = this.state.messages.concat(message);
            this.setState({ messages: joined })
        });

        // Get room and users to update
        this.socket.on('roomUsers', (data) => {
            this.outputUsers(data.users);
        });

    };

    componentDidUpdate() {
        // Scroll chat to bottom
        if (this.bottomMsg.current) {
            this.bottomMsg.current.scrollIntoView({ behaviour: "smooth" });
        }
    }

    componentWillUnmount() {
        // Leave room when off the page
        this.socket.disconnect();
    }

    sendMessage = (event) => {
        event.preventDefault();
        let message = this.inputBox.current.value

        // Ignore empty message
        if (message == "") {
            return;
        }

        // Send message to server
        this.socket.emit("chatMessage",  message, { 
            token: this.state.token
        });

        this.inputBox.current.value = ""

    }

    render() {
        return (
            <div className={styles.Container}>
                <form onSubmit={this.sendMessage}>
                    <Paper style={{maxHeight: 200, height:200, overflow: 'auto'}} >
                        <List className={styles.List} >
                            {this.state.messages.map((message, i) => 
                                <ListItemText key={i} primary={message}></ListItemText>
                            )}
                            <ListItemText ref={this.bottomMsg}></ListItemText>
                        </List>
                    </Paper>
                    <input id="message" className={styles.input} placeholder="Message" ref={this.inputBox} autoComplete="off" />
                    <Button onClick={this.sendMessage} variant="contained" color="primary" >Send</Button>
                </form>
            </div>
        );
    }
}

export default ChatBox;