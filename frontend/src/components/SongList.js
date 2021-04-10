import React, { Component } from 'react';
import axios from 'axios';
import Song from './Song';
import Music from './Music';
import OutsideAlerter from './OutsideAlerter';
import SongCard from './SongCard';

import { List, ListItem, Paper, Grow } from '@material-ui/core/';

class SongList extends Component {
    constructor(props) {
        super(props);
        this.state = {
            songs: [],
            showPopup: false,
            popup_content: "",
            search: "",
            drop: "name",
        };
        this.handleSearchChange = this.handleSearchChange.bind(this);
        this.handleDropChange = this.handleDropChange.bind(this);
    }

    handleClick(song) {
        this.setState({ showPopup: true, popup_content: song });
    }

    handleSearchChange(event) {
        // console.log("search", event.target.value.toLowerCase());
        this.setState({ search: event.target.value.toLowerCase() });
    }

    handleDropChange(event) {
        // console.log("drop", event.target.value.toLowerCase());
        this.setState({ drop: event.target.value.toLowerCase() });
    }

    customSort(a, b) {
        if (a[this.state.drop] == b[this.state.drop]) {
            return (a.artist > b.artist) ? -1 : 1;
        }

        return (a[this.state.drop] > b[this.state.drop]) ? 1 : -1;
    }
    // Runs when component has been added
    componentDidMount() {
        axios.get("/songs")
            .then(res => {
                let data = res.data;
                this.setState({ songs: data });
            }).catch(error => {
                console.log(error);
            });
    };


    removePopup() {
        this.setState({ showPopup: false, popup_content: "" })
    }     

    // What is actually displayed
    // Note: using index as ID can be bad when reordering
    // Note: key changes result in component remount
    render() {
        return (
            <div>
                <div>
                    {this.state.showPopup ?
                        <Grow in={true}>
                            <Paper>
                                <OutsideAlerter removefunc={() => this.removePopup()}>
                                    <SongCard key={this.state.popup_content._id} data={this.state.popup_content}/>
                                    <Music key={"M" + this.state.popup_content._id} />
                                </OutsideAlerter>
                            </Paper>
                        </Grow>
                        :
                        <div>
                            <div>
                                <input type="text" value={this.state.search} onChange={this.handleSearchChange} />
                                <label>
                                    <select value={this.state.drop} onChange={this.handleDropChange}>
                                        <option value="name">Song Title</option>
                                        <option value="artist">Artist</option>
                                        <option value="difficulty">Difficulty</option>
                                        <option value="length">Song Length</option>
                                    </select>
                                </label>
                            </div>
                            <Paper style={{ overflow: 'auto' }} >
                                <List>
                                    {this.state.songs
                                        .filter(song => song.name.toLowerCase().includes(this.state.search) || song.artist.toLowerCase().includes(this.state.search))
                                        .sort((a, b) => this.customSort(a, b))
                                        .map(song =>
                                            <ListItem key={song._id} onClick={() => this.handleClick(song)}> <Song data={song} /> </ListItem>
                                        )
                                    }
                                </List>
                            </Paper>
                        </div>
                    }
                </div>

            </div>
        );
    }
}

export default SongList;