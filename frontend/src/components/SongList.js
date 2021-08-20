import React, { Component } from 'react';
import axios from 'axios';
import Song from './Song';
import OutsideAlerter from './OutsideAlerter';
import SongCard from './SongCard';
import PlayerBox from './PlayerBox';
import LeaderBoard from './LeaderBoard';
import styles from './css/SongList.module.css';
import ImageService from '../services/ImageService';

import UIfx from 'uifx';
import beep from './soundFX/beep.mp3'

import { withRouter } from 'react-router-dom';
import { List, ListItem, Paper, Grow, MenuItem, Select, Fade, Button } from '@material-ui/core/';

// Song list on main page
class SongList extends Component {
    constructor(props) {
        super(props);
        this.state = {
            songs: [],
            showPopup: false,
            popup_content: "",
            search: "",
            selectedSong: "",
            drop: "name",
            token: props.token,
            lobbyID: props.match.params.lobbyID
        };
        
        // beep sound
        this.beep = new UIfx(beep, {
            volume: 0.1
        });

        this.props = props;
        this.handleSearchChange = this.handleSearchChange.bind(this);
        this.handleDropChange = this.handleDropChange.bind(this);
    }

    // Clicking a song
    handleClick(song) {
        this.beep.play();

        // Background in song card, blurring
        if(song == this.state.selectedSong){
            this.style = {
                backgroundImage: "url(" + ImageService.getImagebyID(song._id) + ")",
                filter: "blur(8px)",
                WebkitFilter: "blur(8px)",
                height: "100vh",
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
                backgroundSize: "cover"
            };
            this.setState({ showPopup: true, popup_content: song});
            this.forceUpdate();
        }else{
            this.setState({ selectedSong: song });
        }
    }

    handleSearchChange(event) {
        this.setState({ search: event.target.value.toLowerCase() });
    }

    handleDropChange(event) {
        this.setState({ drop: event.target.value.toLowerCase() });
        this.forceUpdate();
    }

    // Custom list sort when filtering and searching
    customSort(a, b) {
        if (a[this.state.drop] == b[this.state.drop]) {
            return (a.artist > b.artist) ? -1 : 1;
        }
        return (a[this.state.drop] > b[this.state.drop]) ? 1 : -1;
    }

    // Gets list of songs
    componentDidMount() {
        axios.get("/songs",{
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(res => {
                let data = res.data;
                this.setState({ songs: data });
            }).catch(error => {
                console.log(error);
            });
    };

    // Clicking multiplayer button
    // API call to get Multiplayer Lobby ID
    goToMultiplayer = (e) => {
        e.preventDefault();

        axios.post("/createLobby", null, {            
            headers: {
                "Authorization": `Bearer ${this.state.token}`
            }
        }).then(res => {
            this.props.history.push('/multiplayer/' + res.data.lobbyID);
        }).catch(error => {
            console.error(error);
        });
    }

    // Clicking outside song card
    removePopup() {
        this.setState({ showPopup: false, popup_content: "" })
    }

    render() {
        return (
            <div style={{overflowX: "hidden"}}>
                <div>
                    {this.state.showPopup ?
                        <div>
                            <div style={this.style}> </div>
                            <div className={styles.SongCardParentContainer}>
                                <Fade in={true} >
                                    <Paper>
                                        <OutsideAlerter removefunc={() => this.removePopup()}>
                                            <SongCard key={this.state.popup_content._id} data={this.state.popup_content} lobbyID={this.state.lobbyID} />
                                        </OutsideAlerter>
                                    </Paper>
                                </Fade>
                            </div>
                        </div>
                        :
                        <div>
                            <div>
                                <PlayerBox token={this.state.token} />
                                <div className={styles.multiplayer}>
                                    {!this.state.lobbyID && <Button onClick={this.goToMultiplayer}  variant="contained" color="primary">Multiplayer</Button>}
                                </div>
                                
                            </div>
                            <div className={styles.SearchContainer}>
                                <input className={styles.filter} type="text" value={this.state.search} onChange={this.handleSearchChange} />
                                <label>
                                    <Select variant="outlined" className={styles.dropdown} value={this.state.drop} onChange={this.handleDropChange}>
                                        <MenuItem value="name">Song Title</MenuItem>
                                        <MenuItem value="artist">Artist</MenuItem>
                                        <MenuItem value="difficulty">Difficulty</MenuItem>
                                        <MenuItem value="length">Song Length</MenuItem>
                                    </Select>
                                </label>
                            </div>
                            <div className={styles.SongLead}>
                                <div className={styles.LeaderBoardContainer}>
                                    <LeaderBoard token={this.state.token} songID={this.state.selectedSong._id} />
                                </div>
                                <div>
                                    <Paper className={styles.PaperContainer}>
                                        <List className={styles.ListContainer}>
                                            {this.state.songs
                                                .filter(song => song.name.toLowerCase().includes(this.state.search) || song.artist.toLowerCase().includes(this.state.search))
                                                .sort((a, b) => this.customSort(a, b))
                                                .map((song, i) =>
                                                    <Grow key={i} in={true}>
                                                        <ListItem className={styles.ListItem} key={song._id} onClick={() => this.handleClick(song)}> <Song data={song} /> </ListItem>
                                                    </Grow>
                                                )
                                            }
                                        </List>
                                    </Paper>
                                </div>
                            </div>
                        </div>
                    }
                </div>
            </div>
        );
    }
}

export default withRouter(SongList);