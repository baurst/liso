AV2_ALL_CLASSES = [
    "REGULAR_VEHICLE",  # Any conventionally sized passenger vehicle used for the transportation of people and cargo. This includes Cars, vans, pickup trucks, SUVs, etc.
    "PEDESTRIAN",  # Person that is not driving or riding in/on a vehicle. They can be walking, standing, sitting, prone, etc.
    "BOLLARD",  # Bollards are short, sturdy posts installed in the roadway or sidewalk to control the flow of traffic. These may be temporary or permanent and are sometimes decorative.
    "CONSTRUCTION_CONE",  # Movable traffic cone that is used to alert drivers to a hazard. These will typically be orange and white striped and may or may not have a blinking light attached to the top.
    "CONSTRUCTION_BARREL",  # Movable traffic barrel that is used to alert drivers to a hazard. These will typically be orange and white striped and may or may not have a blinking light attached to the top.
    "STOP_SIGN",  # Red octagonal traffic sign displaying the word STOP used to notify drivers that they must come to a complete stop and make sure no other road users are coming before proceeding.
    "BICYCLE",  # Non-motorized vehicle that typically has two wheels and is propelled by human power pushing pedals in a circular motion.
    "LARGE_VEHICLE",  # Large motorized vehicles (four wheels or more) which do not fit into any more specific subclass. Examples include extended passenger vans, fire trucks, RVs, etc.
    "WHEELED_DEVICE",  # Objects involved in the transportation of a person and do not fit a more specific class. Examples range from skateboards, non-motorized scooters, segways, to golf-carts.
    "BUS",  # Standard city buses designed to carry a large number of people.
    "BOX_TRUCK",  # Chassis cab truck with an enclosed cube shaped cargo area. It should be noted that the cargo area is rigidly attached to the cab, and they do not articulate.
    "SIGN",  # Official road signs placed by the Department of Transportation (DOT signs) which are of interest to us. This includes yield signs, speed limit signs, directional control signs, construction signs, and other signs that provide required traffic control information. Note that Stop Sign is captured separately and informative signs such as street signs, parking signs, bus stop signs, etc. are not included in this class.
    "TRUCK",  # Vehicles that are clearly defined as a truck but does not fit into the subclasses of Box Truck or Truck Cab. Examples include common delivery vehicles (UPS, FedEx), mail trucks, garbage trucks, utility trucks, ambulances, dump trucks, etc.
    "MOTORCYCLE",  # Motorized vehicle with two wheels where the rider straddles the engine. These are capable of high speeds similar to a car.
    "BICYCLIST",  # Person actively riding a bicycle, non-pedaling passengers included.
    "VEHICULAR_TRAILER",  # Non-motorized, wheeled vehicle towed behind a motorized vehicle.
    "TRUCK_CAB",  # Heavy truck commonly known as “Semi cab”, “Tractor”, or “Lorry”. This refers to only the front of part of an articulated tractor trailer.
    "MOTORCYCLIST",  # Person actively riding a motorcycle or a moped, including passengers.
    "DOG",  # Any member of the canine family.
    "SCHOOL_BUS",  # Bus that primarily holds school children (typically yellow) and can control the flow of traffic via the use of an articulating stop sign and loading/unloading flasher lights.
    "WHEELED_RIDER",  # Person actively riding or being carried by a wheeled device.
    "STROLLER",  # Push-cart with wheels meant to hold a baby or toddler.
    "ARTICULATED_BUS",  # Articulated buses perform the same function as a standard city bus, but are able to bend (articulate) towards the center. These will also have a third set of wheels not present on a typical bus.
    "MESSAGE_BOARD_TRAILER",  # Trailer carrying a large, mounted, electronic sign to display messages. Often found around construction sites or large events.
    "MOBILE_PEDESTRIAN_SIGN",  # Movable sign designating an area where pedestrians may cross the road.
    "WHEELCHAIR",  # Chair fitted with wheels for use as a means of transport by a person who is unable to walk as a result of illness, injury, or disability. This includes both motorized and non-motorized wheelchairs as well as low-speed seated scooters not intended for use on the roadway.
]

AV2_MOVABLE_CLASSES = [
    "REGULAR_VEHICLE",
    "PEDESTRIAN",
    "BICYCLE",
    "LARGE_VEHICLE",
    "WHEELED_DEVICE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "MOTORCYCLE",
    "BICYCLIST",
    "VEHICULAR_TRAILER",
    "TRUCK_CAB",
    "MOTORCYCLIST",
    "DOG" "SCHOOL_BUS",
    "WHEELED_RIDER",
    "STROLLER",
    "ARTICULATED_BUS",
    "MESSAGE_BOARD_TRAILER",
    "WHEELCHAIR",
]
