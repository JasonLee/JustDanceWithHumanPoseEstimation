import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import ImageService from '../services/ImageService';
import PropTypes from 'prop-types';

export default class SongCard extends Component {
    constructor(props) {
        super(props);
        this.name = props.data.name;
        this.artist = props.data.artist;
        this.length = props.data.length;
        this.creator = props.data.creator;
        this.difficulty = props.data.difficulty;

        this.goToSong = props.startfunc;
        
        this.image = ImageService.getImage("TWICE");
    }

    // What is actually displayed
    render() {
        return ( 
        <div>
            <img src={this.image} />
            <div>
                Name: {this.name}<br />
                Artist: {this.artist}<br />
                Length: {this.length}<br />
                Creator: {this.creator}<br />
                Difficulty: {this.difficulty}<br />
            </div>
            <Link to="/test">
                <button> Start </button>
            </Link>

            <button onClick={() => console.log("Practice Pressed")}> Practise </button>
        </div>
        );
    }
}

SongCard.propTypes = {
    startfunc: PropTypes.func.isRequired
};

