import React, { Component } from 'react';
import { List, ListItem, Paper } from '@material-ui/core/';
import styles from './css/ScoreList.module.css'


export default class ScoreList extends Component {
    constructor(props) {
        super(props);
        
        let prevScores = props.prevScores;

        let cloneScores = JSON.parse(JSON.stringify(prevScores))


        cloneScores.push({score:  props.score, current: true})
        
        this.state = {
            scores: cloneScores,
            currentScore: props.score
        }
        
        
        this.currentScoreRef = React.createRef();
    }
    
    componentDidUpdate() {
        if (this.currentScoreRef.current) {
            this.currentScoreRef.current.scrollIntoView(
                {
                    behavior: 'instant',
                    block: 'center',
                    inline: 'center'
                }
            );
        }
    }

    render() {
        return (
            <div>
                <Paper className={styles.Paper} styles={{ overflow: "auto" }}>
                    <List className={styles.List} styles={{ overflow: "auto" }}>
                        {this.state.scores.sort((a,b) => b.score - a.score).map( (element, i) => { 
                            if(!element.hasOwnProperty('current')) {
                                return <ListItem className={styles.generalItem} key={i}> {i+1}. {element.score} </ListItem> 
                            }else{
                                return <ListItem className={styles.currentItem} key={i} ref={this.currentScoreRef}> YOU: {element.score} </ListItem>
                            }
                        })}
                    </List>
                </Paper>
            </div>
        );
    }
}