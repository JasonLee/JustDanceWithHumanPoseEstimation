import React, { useState } from 'react';
import { Stage, Layer, Line, Circle } from 'react-konva';

export default function ScoreCanvas(props) {
    // console.log(props.joints)
    let joints = props.joints
    let mapping = props.mapping

    console.log("joints", joints);
    // console.log("mapping", mapping);

    if(!mapping || !joints) {
        return (<></>);
    }

    const items = [];
    const circles=[];
    joints.map((joint, i) => {
        if(joint[0] != 0 && joint[1] != 0){
            console.log("joint", i, mapping[i], joints[mapping[i]][0], joints[mapping[i]][1], joints[i][0], joints[i][1]);
            items.push(<Line key={i}
                x={0}
                y={0}
                points={[joints[mapping[i]][0], joints[mapping[i]][1] , joints[i][0], joints[i][1]]}
                stroke="white"
            />);
            circles.push(<Circle x={joint[0]} y={joint[1]} radius={5} fill="green" />) 
        }
        
    })

    return (
        <Stage width={300} height={300}>
            <Layer>
                {items} 
                {circles}
                    
            </Layer>
        </Stage>
    )
};
