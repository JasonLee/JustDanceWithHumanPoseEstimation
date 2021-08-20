import React from 'react';
import { Stage, Layer, Line, Circle } from 'react-konva';

//  Color mappings from results
export function getColorFromScore(score) {
    if(score == 500) {
        return "green"
    }else if(score == 300) {
        return "blue"
    }else if(score == 100) {
            return "gray"
    }else if(score == 0) {
        return "red"
    }else{
        return "black"
    }
}

// Pose box in game
export default function ScoreCanvas(props) {
    const joints = props.joints
    const mapping = props.mapping
    const results = props.results
    const truth = props.truth;
    
    if(!mapping || !joints) {
        return (<div></div>);
    }

    const items = [];
    const circles=[];
    const score = [];

    // Construct joints image
    joints.map((joint, i) => {
        if(i == 0) {
            return;
        }
        if(joint[0] != 0 && joint[1] != 0){
            items.push(<Line key={i}
                x={0}
                y={0}
                points={[joints[mapping[i]][0], joints[mapping[i]][1] , joints[i][0], joints[i][1]]}
                stroke={truth ? "black" : getColorFromScore(results[i])}
            />);

            circles.push(<Circle key={i} x={joint[0]} y={joint[1]} radius={2} fill="green" />) 
        }
        
    })

    return (
        <div className={props.className}>
            <Stage width={250} height={250}>
                <Layer>
                    {items} 
                    {circles}
                    {score}
                </Layer>
            </Stage>
        </div>
    )
};
