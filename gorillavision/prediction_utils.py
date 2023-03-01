def make_tracks(frame_results):
    tracks = {}

    for idx, frame_result in enumerate(frame_results):
        for result in frame_result:
            track_id = result[1]
            item = {
                "frame_idx": idx,
                "bbox": result[2:5],
                "class": result[0],
            }
            if track_id not in tracks:
                tracks[track_id] = [item]
            else:
                tracks[track_id].append(item)

    return tracks

def join_tracks(tracks, identitites):
    new_tracks = {}
    for identity in identitites.keys():
        if identity not in new_tracks:
            new_tracks[identity] = tracks[identitites[identity]]
        else:
            new_tracks[identity] += tracks[identitites[identity]]
    
    return new_tracks

    