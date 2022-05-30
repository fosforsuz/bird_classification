package com.example.meyvesinif;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.meyvesinif.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 64;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 64, 64, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 1));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"ABBOTTS BABBLER", "ABBOTTS BOOBY", "ABYSSINIAN GROUND HORNBILL",
                    "AFRICAN CROWNED CRANE", "AFRICAN EMERALD CUCKOO", "AFRICAN FIREFINCH", "AFRICAN OYSTER CATCHER",
                    "ALBATROSS", "ALBERTS TOWHEE", "ALEXANDRINE PARAKEET", "ALPINE CHOUGH", "ALTAMIRA YELLOWTHROAT", "AMERICAN AVOCET",
                    "AMERICAN BITTERN", "AMERICAN COOT", "AMERICAN GOLDFINCH", "AMERICAN KESTREL", "AMERICAN PIPIT", "AMERICAN REDSTART",
                    "AMETHYST WOODSTAR", "ANDEAN GOOSE", "ANDEAN LAPWING", "ANDEAN SISKIN", "ANHINGA", "ANIANIAU", "ANNAS HUMMINGBIRD",
                    "ANTBIRD", "ANTILLEAN EUPHONIA", "APAPANE", "APOSTLEBIRD", "ARARIPE MANAKIN", "ASHY THRUSHBIRD", "ASIAN CRESTED IBIS",
                    "AVADAVAT", "AZURE JAY", "AZURE TANAGER", "AZURE TIT", "BAIKAL TEAL", "BALD EAGLE", "BALD IBIS", "BALI STARLING",
                    "BALTIMORE ORIOLE", "BANANAQUIT", "BAND TAILED GUAN", "BANDED BROADBILL", "BANDED PITA", "BANDED STILT",
                    "BAR-TAILED GODWIT", "BARN OWL", "BARN SWALLOW", "BARRED PUFFBIRD", "BARROWS GOLDENEYE", "BAY-BREASTED WARBLER",
                    "BEARDED BARBET", "BEARDED BELLBIRD", "BEARDED REEDLING", "BELTED KINGFISHER", "BIRD OF PARADISE", "BLACK & YELLOW  BROADBILL",
                    "BLACK BAZA", "BLACK COCKATO", "BLACK FRANCOLIN", "BLACK SKIMMER", "BLACK SWAN", "BLACK TAIL CRAKE", "BLACK THROATED BUSHTIT",
                    "BLACK THROATED WARBLER", "BLACK VULTURE", "BLACK-CAPPED CHICKADEE", "BLACK-NECKED GREBE", "BLACK-THROATED SPARROW",
                    "BLACKBURNIAM WARBLER", "BLONDE CRESTED WOODPECKER", "BLUE COAU", "BLUE GROUSE", "BLUE HERON", "BLUE THROATED TOUCANET",
                    "BOBOLINK", "BORNEAN BRISTLEHEAD", "BORNEAN LEAFBIRD", "BORNEAN PHEASANT", "BRANDT CORMARANT", "BROWN CREPPER", "BROWN NOODY",
                    "BROWN THRASHER", "BULWERS PHEASANT", "BUSH TURKEY", "CACTUS WREN", "CALIFORNIA CONDOR", "CALIFORNIA GULL", "CALIFORNIA QUAIL",
                    "CANARY", "CAPE GLOSSY STARLING", "CAPE LONGCLAW", "CAPE MAY WARBLER", "CAPE ROCK THRUSH", "CAPPED HERON", "CAPUCHINBIRD",
                    "CARMINE BEE-EATER", "CASPIAN TERN", "CASSOWARY", "CEDAR WAXWING", "CERULEAN WARBLER", "CHARA DE COLLAR", "CHATTERING LORY",
                    "CHESTNET BELLIED EUPHONIA", "CHINESE BAMBOO PARTRIDGE", "CHINESE POND HERON", "CHIPPING SPARROW", "CHUCAO TAPACULO",
                    "CHUKAR PARTRIDGE", "CINNAMON ATTILA", "CINNAMON FLYCATCHER", "CINNAMON TEAL", "CLARKS NUTCRACKER", "COCK OF THE  ROCK",
                    "COCKATOO", "COLLARED ARACARI", "COMMON FIRECREST", "COMMON GRACKLE", "COMMON HOUSE MARTIN", "COMMON IORA", "COMMON LOON",
                    "COMMON POORWILL", "COMMON STARLING", "COPPERY TAILED COUCAL", "CRAB PLOVER", "CRANE HAWK", "CREAM COLORED WOODPECKER",
                    "CRESTED AUKLET", "CRESTED CARACARA", "CRESTED COUA", "CRESTED FIREBACK", "CRESTED KINGFISHER", "CRESTED NUTHATCH",
                    "CRESTED OROPENDOLA", "CRESTED SHRIKETIT", "CRIMSON CHAT", "CRIMSON SUNBIRD", "CROW", "CROWNED PIGEON", "CUBAN TODY",
                    "CUBAN TROGON", "CURL CRESTED ARACURI", "D-ARNAUDS BARBET", "DARK EYED JUNCO", "DEMOISELLE CRANE", "DOUBLE BARRED FINCH",
                    "DOUBLE BRESTED CORMARANT", "DOUBLE EYED FIG PARROT", "DOWNY WOODPECKER", "DUSKY LORY", "EARED PITA", "EASTERN BLUEBIRD",
                    "EASTERN GOLDEN WEAVER", "EASTERN MEADOWLARK", "EASTERN ROSELLA", "EASTERN TOWEE", "ELEGANT TROGON", "ELLIOTS  PHEASANT",
                    "EMERALD TANAGER", "EMPEROR PENGUIN", "EMU", "ENGGANO MYNA", "EURASIAN GOLDEN ORIOLE", "EURASIAN MAGPIE", "EUROPEAN GOLDFINCH",
                    "EUROPEAN TURTLE DOVE", "EVENING GROSBEAK", "FAIRY BLUEBIRD", "FAIRY TERN", "FIORDLAND PENGUIN", "FIRE TAILLED MYZORNIS",
                    "FLAME BOWERBIRD", "FLAME TANAGER", "FLAMINGO", "FRIGATE", "GAMBELS QUAIL", "GANG GANG COCKATOO", "GILA WOODPECKER",
                    "GILDED FLICKER", "GLOSSY IBIS", "GO AWAY BIRD", "GOLD WING WARBLER", "GOLDEN CHEEKED WARBLER", "GOLDEN CHLOROPHONIA",
                    "GOLDEN EAGLE", "GOLDEN PHEASANT", "GOLDEN PIPIT", "GOULDIAN FINCH", "GRAY CATBIRD", "GRAY KINGBIRD", "GRAY PARTRIDGE",
                    "GREAT GRAY OWL", "GREAT JACAMAR", "GREAT KISKADEE", "GREAT POTOO", "GREATOR SAGE GROUSE", "GREEN BROADBILL", "GREEN JAY",
                    "GREEN MAGPIE", "GREY PLOVER", "GROVED BILLED ANI", "GUINEA TURACO", "GUINEAFOWL", "GURNEYS PITTA", "GYRFALCON", "HAMMERKOP",
                    "HARLEQUIN DUCK", "HARLEQUIN QUAIL", "HARPY EAGLE", "HAWAIIAN GOOSE", "HAWFINCH", "HELMET VANGA", "HEPATIC TANAGER",
                    "HIMALAYAN BLUETAIL", "HIMALAYAN MONAL", "HOATZIN", "HOODED MERGANSER", "HOOPOES", "HORNBILL", "HORNED GUAN", "HORNED LARK",
                    "HORNED SUNGEM", "HOUSE FINCH", "HOUSE SPARROW", "HYACINTH MACAW", "IBERIAN MAGPIE", "IBISBILL", "IMPERIAL SHAQ", "INCA TERN",
                    "INDIAN BUSTARD", "INDIAN PITTA", "INDIAN ROLLER", "INDIGO BUNTING", "INLAND DOTTEREL", "IVORY GULL", "IWI", "JABIRU",
                    "JACK SNIPE", "JANDAYA PARAKEET", "JAPANESE ROBIN", "JAVA SPARROW", "KAGU", "KAKAPO", "KILLDEAR", "KING VULTURE", "KIWI",
                    "KOOKABURRA", "LARK BUNTING", "LAZULI BUNTING", "LESSER ADJUTANT", "LILAC ROLLER", "LITTLE AUK", "LONG-EARED OWL", "MAGPIE GOOSE",
                    "MALABAR HORNBILL", "MALACHITE KINGFISHER", "MALAGASY WHITE EYE", "MALEO", "MALLARD DUCK", "MANDRIN DUCK", "MANGROVE CUCKOO",
                    "MARABOU STORK", "MASKED BOOBY", "MASKED LAPWING", "MIKADO  PHEASANT", "MOURNING DOVE", "MYNA", "NICOBAR PIGEON", "NOISY FRIARBIRD",
                    "NORTHERN CARDINAL", "NORTHERN FLICKER", "NORTHERN FULMAR", "NORTHERN GANNET", "NORTHERN GOSHAWK", "NORTHERN JACANA",
                    "NORTHERN MOCKINGBIRD", "NORTHERN PARULA", "NORTHERN RED BISHOP", "NORTHERN SHOVELER", "OCELLATED TURKEY", "OKINAWA RAIL",
                    "ORANGE BRESTED BUNTING", "ORIENTAL BAY OWL", "OSPREY", "OSTRICH", "OVENBIRD", "OYSTER CATCHER", "PAINTED BUNTING",
                    "PALILA", "PARADISE TANAGER", "PARAKETT  AKULET", "PARUS MAJOR", "PATAGONIAN SIERRA FINCH", "PEACOCK", "PELICAN",
                    "PEREGRINE FALCON", "PHILIPPINE EAGLE", "PINK ROBIN", "POMARINE JAEGER", "PUFFIN", "PURPLE FINCH", "PURPLE GALLINULE",
                    "PURPLE MARTIN", "PURPLE SWAMPHEN", "PYGMY KINGFISHER", "QUETZAL", "RAINBOW LORIKEET", "RAZORBILL", "RED BEARDED BEE EATER",
                    "RED BELLIED PITTA", "RED BROWED FINCH", "RED FACED CORMORANT", "RED FACED WARBLER", "RED FODY", "RED HEADED DUCK",
                    "RED HEADED WOODPECKER", "RED HONEY CREEPER", "RED NAPED TROGON", "RED TAILED HAWK", "RED TAILED THRUSH",
                    "RED WINGED BLACKBIRD", "RED WISKERED BULBUL", "REGENT BOWERBIRD", "RING-NECKED PHEASANT", "ROADRUNNER", "ROBIN",
                    "ROCK DOVE", "ROSY FACED LOVEBIRD", "ROUGH LEG BUZZARD", "ROYAL FLYCATCHER", "RUBY THROATED HUMMINGBIRD",
                    "RUDY KINGFISHER", "RUFOUS KINGFISHER", "RUFUOS MOTMOT", "SAMATRAN THRUSH", "SAND MARTIN", "SANDHILL CRANE",
                    "SATYR TRAGOPAN", "SCARLET CROWNED FRUIT DOVE", "SCARLET IBIS", "SCARLET MACAW", "SCARLET TANAGER", "SHOEBILL",
                    "SHORT BILLED DOWITCHER", "SMITHS LONGSPUR", "SNOWY EGRET", "SNOWY OWL", "SORA", "SPANGLED COTINGA", "SPLENDID WREN",
                    "SPOON BILED SANDPIPER", "SPOONBILL", "SPOTTED CATBIRD", "SRI LANKA BLUE MAGPIE", "STEAMER DUCK", "STORK BILLED KINGFISHER",
                    "STRAWBERRY FINCH", "STRIPED OWL", "STRIPPED MANAKIN", "STRIPPED SWALLOW", "SUPERB STARLING", "SWINHOES PHEASANT", "TAILORBIRD",
                    "TAIWAN MAGPIE", "TAKAHE", "TASMANIAN HEN", "TEAL DUCK", "TIT MOUSE", "TOUCHAN", "TOWNSENDS WARBLER", "TREE SWALLOW",
                    "TROPICAL KINGBIRD", "TRUMPTER SWAN", "TURKEY VULTURE", "TURQUOISE MOTMOT", "UMBRELLA BIRD", "VARIED THRUSH", "VENEZUELIAN TROUPIAL",
                    "VERMILION FLYCATHER", "VICTORIA CROWNED PIGEON", "VIOLET GREEN SWALLOW", "VIOLET TURACO", "VULTURINE GUINEAFOWL", "WALL CREAPER",
                    "WATTLED CURASSOW", "WATTLED LAPWING", "WHIMBREL", "WHITE BROWED CRAKE", "WHITE CHEEKED TURACO", "WHITE NECKED RAVEN",
                    "WHITE TAILED TROPIC", "WHITE THROATED BEE EATER", "WILD TURKEY", "WILSONS BIRD OF PARADISE", "WOOD DUCK",
                    "YELLOW BELLIED FLOWERPECKER", "YELLOW CACIQUE", "YELLOW HEADED BLACKBIRD"};
            result.setText(String.valueOf(classes[maxPos]));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}