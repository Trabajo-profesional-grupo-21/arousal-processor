function [ total_bboxes ] = apply_correction( total_bboxes, corrections, add1 )
%APPLY_CORRECTION Summary of this function goes here
%   Detailed explanation goes here

    % Perform correction based on regression values
    bbw = total_bboxes(:,3) - total_bboxes(:,1);
    bbh = total_bboxes(:,4) - total_bboxes(:,2);
    
    % TODO is this needed?
    if(add1)
        bbw = bbw + 1;
        bbh = bbh + 1;
    end
    
    new_min_x = total_bboxes(:,1) + corrections(:,1) .* bbw;
    new_min_y = total_bboxes(:,2) + corrections(:,2) .* bbh;    
    new_max_x = total_bboxes(:,3) + corrections(:,3) .* bbw;
    new_max_y = total_bboxes(:,4) + corrections(:,4) .* bbh;
    score = total_bboxes(:,5);
    total_bboxes = [new_min_x, new_min_y, new_max_x, new_max_y, score];

end

