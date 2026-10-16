"""Native hand/object contact telemetry, separate from original task gates.

Reads post-refreshed MuJoCo state under the last held ctrl. This is not the
constraint force used inside the preceding integration step. No forward/step,
state writes, force injection or collision-model changes occur here.
"""

import math


def finger_group(body_name, geom_name):
    for name in ("thumb", "index", "middle"):
        if body_name.startswith("right_hand_" + name + "_"):
            return name
    return "palm" if geom_name == "right_palm_contact_pad" else None


def summarize_contacts(contacts):
    groups = {name: {"geometric_contacts": 0, "loaded_contacts": 0,
        "normal_force_n": 0., "maximum_penetration_m": 0.,
        "maximum_loaded_penetration_m": 0.} for name in ("thumb", "index", "middle", "palm")}
    for row in contacts:
        if row["group"] not in groups:
            continue
        group = groups[row["group"]]
        depth = max(0., -row["distance_m"])
        group["maximum_penetration_m"] = max(group["maximum_penetration_m"], depth)
        if row["distance_m"] <= 0.:
            group["geometric_contacts"] += 1
        # A positive margin contact is retained in raw telemetry but is not
        # silently counted as surface contact or load-bearing closure.
        if row["distance_m"] <= 0. and row["efc_address"] >= 0 and row["force_contact_frame"][0] > 0.:
            group["loaded_contacts"] += 1
            group["normal_force_n"] += row["force_contact_frame"][0]
            group["maximum_loaded_penetration_m"] = max(group["maximum_loaded_penetration_m"], depth)
    return groups


def native_hand_contacts(model, data, object_body, api):
    import numpy as np
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        gids = (int(contact.geom1), int(contact.geom2))
        bids = tuple(int(model.geom_bodyid[g]) for g in gids)
        if object_body not in bids:
            continue
        other = 1 - bids.index(object_body)
        body = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, bids[other]) or ""
        geom = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, gids[other]) or "geom#{}".format(gids[other])
        group = finger_group(body, geom)
        if group is None:
            continue
        force = np.zeros(6)
        api.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all() or not math.isfinite(float(contact.dist)):
            raise ValueError("nonfinite native hand contact telemetry")
        contacts.append({"group": group, "geom_ids": list(gids), "hand_geom": geom,
            "hand_geom_type": int(model.geom_type[gids[other]]), "object_geom_index": bids.index(object_body),
            "distance_m": float(contact.dist), "includemargin_m": float(contact.includemargin),
            "efc_address": int(contact.efc_address), "dimension": int(contact.dim),
            "force_contact_frame": force.tolist(), "frame_rows_world": np.asarray(contact.frame).reshape(3, 3).tolist(),
            "position_world_m": np.asarray(contact.pos).tolist()})
    return {"timestamp_s": float(data.time), "groups": summarize_contacts(contacts), "contacts": contacts,
            "held_ctrl": np.asarray(data.ctrl).tolist(),
            "actuator_force": np.asarray(data.actuator_force).tolist(),
            "force_sample_scope": "post_mj_step_existing_mj_forward_under_held_ctrl; not integration solver force"}
