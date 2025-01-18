use burn::data::dataset::{Dataset, HuggingfaceDatasetLoader, SqliteDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};

/// String that is passed to the model
#[derive(new, Clone, Debug)]
pub struct TextGenerationItem {
    /// The string that is passed to the model
    pub text: String,
}

/// A single item from [DbPediaDataset]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DbPediaItem {
    /// The content of the item
    pub content: String,
}

/// The dbpedia_14 dataset
///
/// https://huggingface.co/datasets/dbpedia_14
pub struct DbPediaDataset {
    /// Dataset of [DbPediaItem]s converted into [SqliteDataset]
    dataset: SqliteDataset<DbPediaItem>,
}

impl Dataset<TextGenerationItem> for DbPediaDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.content))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DbPediaDataset {
    /// Constructs a new dataset from the given split
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }

    /// Alias for dataset with split "train"
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Alias for dataset with split "test"
    pub fn test() -> Self {
        Self::new("test")
    }
}

/// A single item of [DvachDataset]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DvachItem {
    /// Conversation in the vec of strings for each message
    pub conversation: Vec<String>,
}

/// The 2ch.hk dataset
///
/// https://huggingface.co/datasets/Vikhrmodels/2ch-24-09-2024-no-links
pub struct DvachDataset {
    /// Dataset of [DvachItem]s converted into [SqliteDataset]
    pub dataset: SqliteDataset<DvachItem>,
}

impl DvachDataset {
    /// Constructs a new dataset from the given split
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<DvachItem> =
            HuggingfaceDatasetLoader::new("Vikhrmodels/2ch-24-09-2024-no-links")
                .dataset(split)
                .unwrap();

        Self { dataset }
    }

    /// Alias for train dataset
    pub fn train() -> Self {
        Self::new("train")
    }
}

impl Dataset<TextGenerationItem> for DvachDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem::new(item.conversation.join(" ")))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
