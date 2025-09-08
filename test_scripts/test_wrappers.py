from environments.old.ARSL_env_camera_2 import MicrorobotEnvRayWrapped

def get_env_size():
    env = MicrorobotEnvRayWrapped()
    env.reset()
    _, _, _, _ = env.step(0)
    size = env.observation_space.shape
    return size


if __name__ == "__main__":
    print(get_env_size())