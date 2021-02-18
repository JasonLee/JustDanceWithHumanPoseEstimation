import React, { Component } from 'react';
import ImageService from '../services/ImageService';


class Song extends Component {
    constructor(props) {
        super(props);
        this.name = props.data.name;
        this.artist = props.data.artist;
                
        this.image = ImageService.getImage("TWICE");
    }

    // What is actually displayed
    render() {
        return ( 
        <div>
            <img src={this.image} />
            <div>
                {this.name}<br />
                {this.artist}
            </div>
        </div>
        );
    }
}

export default Song;