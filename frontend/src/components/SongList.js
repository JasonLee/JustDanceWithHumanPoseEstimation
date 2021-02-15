import React, { Component } from 'react';
import axios from 'axios';
import Song from './Song';
import OutsideAlerter from './OutsideAlerter';
import SongCard from './SongCard';
import { List, ListItem, Paper } from '@material-ui/core/';

class SongList extends Component {
    constructor(props) {
        super(props);
        this.state  = {
            songs: [],
            showPopup: false,
            popup_content: ""
        };
    }

    handleClick(song) {
        this.setState({showPopup: true, popup_content: song})
        console.log(song);
    }

    // Runs when component has been added
    componentDidMount() {
        axios.get("/songs")
        .then(res => {
            let data = res.data;
            this.setState({songs: data});
        })
    };

    removePopup() {
        this.setState({showPopup: false, popup_content: ""})
    }

    // What is actually displayed
    // Note: using index as ID can be bad when reordering
    // Note: key changes result in component remount
    render() {
        return ( 
            <div>
                <div>
                    {this.state.showPopup ? 
                        <OutsideAlerter removefunc={() => this.removePopup()}>
                            <SongCard key={this.state.popup_content._id}  data={this.state.popup_content}/>
                        </OutsideAlerter>
                    :   <Paper style={{overflow: 'auto'}} >
                            <List>
                                {this.state.songs.map(song=> 
                                    <ListItem key={song._id}  onClick={() => this.handleClick(song)} ><Song data={song} /></ListItem>
                                )}
                            </List>
                        </Paper>
                    }
                </div>

            </div>
        );
    }
}

export default SongList;