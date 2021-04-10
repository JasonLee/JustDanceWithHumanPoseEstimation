import React, { useState } from 'react';
import { Stage, Layer, Line, Circle } from 'react-konva';


export function getColorFromScore(score) {
    if(score == 500) {
        return "green"
    }else if(score == 300) {
        return "blue"
    }else if(score == 100) {
            return "yellow"
    }else if(score == 0) {
        return "red"
    }else{
        return "white"
    }
}

export default function ScoreCanvas(props) {
    const joints = props.joints
    const mapping = props.mapping
    const results = props.results

    // console.log("joints", joints);
    // console.log("mapping", mapping);

    if(!mapping || !joints) {
        return (<></>);
    }

    const items = [];
    const circles=[];
    joints.map((joint, i) => {
        if(joint[0] != 0 && joint[1] != 0){
            // console.log("joint", i, mapping[i], joints[mapping[i]][0], joints[mapping[i]][1], joints[i][0], joints[i][1]);
            items.push(<Line key={i}
                x={0}
                y={0}
                points={[joints[mapping[i]][0], joints[mapping[i]][1] , joints[i][0], joints[i][1]]}
                stroke={getColorFromScore(results[i])}
            />);
            circles.push(<Circle key={i} x={joint[0]} y={joint[1]} radius={2} fill="green" />) 
        }
        
    })

    return (
        <Stage width={500} height={500}>
            <Layer>
                {items} 
                {circles}
                    
            </Layer>
        </Stage>
    )
};
