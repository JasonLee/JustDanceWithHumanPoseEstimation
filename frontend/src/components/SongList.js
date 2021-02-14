import React, { Component } from 'react';
import axios from 'axios';
import Song from './Song';

class SongList extends Component {
    constructor(props) {
        super(props);
        this.state  = {
            songs: []
        };
    }

    // Runs when component has been added
    componentDidMount() {
        axios.get("/test")
        .then(res => {
            let data = res.data;
            console.log(data);

            // data.sort((a,b) => {
            //     return a.info.game_datetime - b.info.game_datetime;
            // })
            
            this.setState({songs: data});

        })
    };

    // What is actually displayed
    render() {
        return ( 
        <div>
            <div>
                <ol>
                    {this.state.songs.map(song => 
                        <div><Song id={song.id} name={song.name} artist={song.artist}/></div>
                    )}
                </ol>
            </div>
        </div>
        );
    }
}

export default SongList;