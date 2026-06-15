def truncate_on_timeout(env, max_frames=300):
    return env.frame_iteration >= max_frames
