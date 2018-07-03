

def send_to_simulator(sim, env_str, tree, near=2.0, far=8.0, radius=1.0, height=1.0):

    leaves = tree.get_leaf_pos_and_dir()
    assert len(env_str) == len(leaves), '# of objects does not equal # of leaves'

    sensor_ids = []

    for leaf, cyl in zip(leaves, env_str):
        if cyl == '0':
            dist = near
            r, g, b = 1, 0, 0
        if cyl == '1':
            dist = far
            r, g, b = 0, 0, 1
            
        x = leaf['pos'][0] + dist * leaf['dir'][0]
        y = leaf['pos'][1] + dist * leaf['dir'][1]
        # z = leaf['pos'][2] + dist * leaf['dir'][2]

        cyl_id = sim.send_cylinder(x=x, y=y, z=height / 2.0,
                          r1=0, r2=0, r3=1,
                          length=height,
                          radius=radius,
                          r=r, g=g, b=b,
                          capped=False
                          )
        sensor_ids.append(sim.send_is_seen_sensor(cyl_id))


    return sensor_ids